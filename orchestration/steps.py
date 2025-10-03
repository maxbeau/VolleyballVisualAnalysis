"""
Step definitions for the main orchestration, refactored into classes.
"""
from __future__ import annotations
import logging
from typing import Set

from core.cache import detection_cache_dir
from core.context import PipelineContext
from core.steps import OrchestrationStep
from court.homography import compute_and_save_homography, generate_birdseye_image
from court.processing import run_tracking
from detection.pipeline import DetectionPipeline
from players.track import TrackerConfig, run_player_tracking
from ball.ball_trajectory import (
    run_trajectory_analysis as run_ball_trajectory_analysis,
    TrajectoryIOConfig,
    TrajectoryAnalysisConfig,
)
from ball.trajectory_segmentation import TrajectorySegmentationConfig
from visualization.overlay import run_overlay as run_overlay_processing


class DetectionStep(OrchestrationStep):
    """An orchestration step that runs object detection for a specific target."""

    def __init__(self, context: PipelineContext, target: str):
        super().__init__(context)
        self.target = target

    @property
    def name(self) -> str:
        return f"detection_{self.target}"

    def run(self) -> None:
        detection_settings = self.settings.detection
        global_settings = self.settings.global_settings

        model_id = detection_settings.models_roboflow.get(self.target)
        if not model_id:
            raise ValueError(f"Model ID for target '{self.target}' not found in config")

        target_cache_dir = detection_cache_dir(
            global_settings.video_path, global_settings.cache_dir, self.target
        )
        
        output_filename = f"{self.target}_detections.jsonl"
        combined_jsonl = self.context.output_dir / output_filename

        pipeline = DetectionPipeline(
            video_path=str(global_settings.video_path.resolve()),
            target_name=self.target,
            model_id=model_id,
            confidence=global_settings.min_confidence,
            fps_sample=detection_settings.infer_fps.get(self.target, 1),
            cache_dir=str(target_cache_dir.resolve()),
            combined_jsonl=str(combined_jsonl.resolve()),
            save_frame_json=True,
            detection_settings=detection_settings.model_dump(exclude={"roboflow_api_key"}),
        )
        pipeline.run()
        
        self.context.register_artifact(f"{self.target}_detections", output_filename)


class CourtProcessingStep(OrchestrationStep):
    """Runs court tracking and orientation analysis."""

    @property
    def name(self) -> str:
        return "court_processing"

    @property
    def dependencies(self) -> Set[str]:
        return {"detection_court"}

    def run(self) -> None:
        court_settings = self.settings.court
        detections_jsonl = self.context.get_artifact_path("court_detections")

        tracking_jsonl = self.context.output_dir / court_settings.outputs.output_tracking_jsonl
        tracking_meta_json = self.context.output_dir / court_settings.outputs.output_meta_json

        run_tracking(
            video_path=str(self.settings.global_settings.video_path.resolve()),
            detections_jsonl=str(detections_jsonl.resolve()),
            tracking_jsonl=str(tracking_jsonl.resolve()),
            tracking_meta_json=str(tracking_meta_json.resolve()),
            cfg=court_settings,
        )
        
        self.context.register_artifact("court_tracking", court_settings.outputs.output_tracking_jsonl)
        self.context.register_artifact("court_meta", court_settings.outputs.output_meta_json)


class CourtHomographyStep(OrchestrationStep):
    """Computes court homography and generates a bird's-eye view image."""

    @property
    def name(self) -> str:
        return "court_homography"

    @property
    def dependencies(self) -> Set[str]:
        return {"court_processing"}

    def run(self) -> None:
        court_settings = self.settings.court
        tracking_jsonl = self.context.get_artifact_path("court_tracking")
        
        output_npy = self.context.output_dir / court_settings.outputs.output_homography_npy
        output_meta_json = self.context.get_artifact_path("court_meta") # Use existing meta file
        output_jpg = self.context.output_dir / court_settings.outputs.output_birdseye_jpg

        model_size = (
            (court_settings.homography.model_width, court_settings.homography.model_height)
            if court_settings.homography.model_width and court_settings.homography.model_height
            else None
        )

        H, dst_size = compute_and_save_homography(
            tracking_jsonl=str(tracking_jsonl.resolve()),
            output_npy=str(output_npy.resolve()),
            output_meta_json=str(output_meta_json.resolve()),
            model_size=model_size,
            scale_px_per_meter=court_settings.homography.scale_px_per_meter,
        )

        generate_birdseye_image(
            video_path=str(self.settings.global_settings.video_path.resolve()),
            H=H,
            dst_size=dst_size,
            output_jpg=str(output_jpg.resolve()),
            frame_index=court_settings.homography.birdseye_frame_index,
        )
        
        self.context.register_artifact("homography_matrix", court_settings.outputs.output_homography_npy)
        self.context.register_artifact("birdseye_image", court_settings.outputs.output_birdseye_jpg)


class PlayersTrackingStep(OrchestrationStep):
    """Runs player tracking."""

    @property
    def name(self) -> str:
        return "players_tracking"

    @property
    def dependencies(self) -> Set[str]:
        return {"detection_players", "court_processing"}

    def run(self) -> None:
        player_settings = self.settings.players
        detections_jsonl = self.context.get_artifact_path("players_detections")
        court_tracking_jsonl = self.context.get_artifact_path("court_tracking")

        cfg = TrackerConfig(**player_settings.tracker_kwargs())
        tracks_jsonl = self.context.output_dir / player_settings.output_tracks_jsonl

        run_player_tracking(
            video_path=str(self.settings.global_settings.video_path.resolve()),
            detections_jsonl=str(detections_jsonl.resolve()),
            tracks_jsonl=str(tracks_jsonl.resolve()),
            cfg=cfg,
            court_tracking_jsonl=str(court_tracking_jsonl.resolve()),
            max_frames=player_settings.max_frames or 0,
        )
        
        self.context.register_artifact("players_tracks", player_settings.output_tracks_jsonl)


class TrajectoryAnalysisStep(OrchestrationStep):
    """Runs ball trajectory analysis."""

    @property
    def name(self) -> str:
        return "trajectory_analysis"

    @property
    def dependencies(self) -> Set[str]:
        return {"detection_ball", "court_homography"}

    def run(self) -> None:
        traj_settings = self.settings.trajectory_analysis
        detections_jsonl = self.context.get_artifact_path("ball_detections")
        homography_npy = self.context.get_artifact_path("homography_matrix")
        homography_meta_json = self.context.get_artifact_path("court_meta")
        birdseye_jpg = self.context.get_artifact_path("birdseye_image")

        output_jsonl = self.context.output_dir / traj_settings.output_jsonl
        output_csv = self.context.output_dir / traj_settings.output_csv
        output_path_img = self.context.output_dir / traj_settings.output_path_img
        output_segments_img = None
        if getattr(traj_settings, "output_segments_img", None):
            output_segments_img = self.context.output_dir / traj_settings.output_segments_img

        io_cfg = TrajectoryIOConfig(
            video_path=str(self.settings.global_settings.video_path.resolve()),
            detections_jsonl=str(detections_jsonl.resolve()),
            homography_npy=str(homography_npy.resolve()),
            homography_meta_json=str(homography_meta_json.resolve()),
            birdseye_jpg=str(birdseye_jpg.resolve()),
            output_jsonl=str(output_jsonl.resolve()),
            output_csv=str(output_csv.resolve()),
            output_path_img=str(output_path_img.resolve()),
            output_segmentation_img=str(output_segments_img.resolve()) if output_segments_img else None,
        )

        analysis_cfg = TrajectoryAnalysisConfig(
            min_confidence=self.settings.global_settings.min_confidence,
            ar_filter_min=traj_settings.ar_filter_min,
            ar_filter_max=traj_settings.ar_filter_max,
            ar_filter_alpha=traj_settings.ar_filter_alpha,
            max_interp_gap=traj_settings.max_interp_gap,
            obs_gate_chisq=traj_settings.obs_gate_chisq,
            obs_gate_use_conf=traj_settings.obs_gate_use_conf,
            hold_ttl=traj_settings.hold_ttl,
            max_speed_px_per_frame=traj_settings.max_speed_px_per_frame,
            max_accel_px_per_frame2=traj_settings.max_accel_px_per_frame2,
            speed_reset_frame_gap=traj_settings.speed_reset_frames,
            static_filter_enable=traj_settings.static_filter_enable,
            static_window_frames=traj_settings.static_window,
            static_min_motion_px=traj_settings.static_min_motion_px,
            continuity_filter_enable=traj_settings.continuity_filter_enable,
            continuity_window_frames=traj_settings.continuity_window,
            continuity_max_error_px=traj_settings.continuity_max_error_px,
            continuity_error_growth_px=traj_settings.continuity_error_growth_px,
            viterbi_cfg=traj_settings.viterbi.model_dump(),
            ball_diameter_m=traj_settings.ball_diameter_m,
            size_model_min_samples=traj_settings.size_model_min_samples,
            measurement_confidence_floor=traj_settings.measurement_confidence_floor,
            height_max_m=traj_settings.height_max_m,
            segmentation=TrajectorySegmentationConfig(**traj_settings.segmentation.model_dump()),
        )

        run_ball_trajectory_analysis(io_cfg, analysis_cfg)
        
        self.context.register_artifact("ball_trajectory_jsonl", traj_settings.output_jsonl)
        self.context.register_artifact("ball_trajectory_csv", traj_settings.output_csv)
        self.context.register_artifact("ball_path_image", traj_settings.output_path_img)
        if output_segments_img:
            self.context.register_artifact("ball_segments_image", traj_settings.output_segments_img)


class OverlayStep(OrchestrationStep):
    """Generates the final overlay video."""

    @property
    def name(self) -> str:
        return "overlay"

    @property
    def dependencies(self) -> Set[str]:
        deps: Set[str] = {"trajectory_analysis", "court_processing"}
        if self.settings.overlay.players.enable:
            deps.add("players_tracking")
        if self.settings.overlay.actions.enable:
            deps.add("detection_actions")
        return deps

    def run(self) -> None:
        overlay_settings = self.settings.overlay
        
        ball_trajectory_jsonl = self.context.get_artifact_path("ball_trajectory_jsonl")
        court_tracking_jsonl = self.context.get_artifact_path("court_tracking")
        court_tracking_meta_json = self.context.get_artifact_path("court_meta")

        players_tracks_jsonl = None
        if overlay_settings.players.enable:
            players_tracks_jsonl = self.context.get_artifact_path("players_tracks")

        actions_detections_jsonl = None
        if overlay_settings.actions.enable:
            actions_detections_jsonl = self.context.get_artifact_path("actions_detections")
        
        actions_clips_jsonl = self.context.output_dir / overlay_settings.actions.clips_jsonl

        cfg = overlay_settings.model_dump()

        run_overlay_processing(
            cfg=cfg,
            video_path=str(self.settings.global_settings.video_path.resolve()),
            ball_trajectory_jsonl=str(ball_trajectory_jsonl.resolve()),
            court_tracking_jsonl=str(court_tracking_jsonl.resolve()),
            players_tracks_jsonl=str(players_tracks_jsonl.resolve()) if players_tracks_jsonl else None,
            actions_detections_jsonl=str(actions_detections_jsonl.resolve()) if actions_detections_jsonl else None,
            actions_clips_jsonl=str(actions_clips_jsonl.resolve()),
            court_tracking_meta_json=str(court_tracking_meta_json.resolve()),
        )
        
        self.context.register_artifact("overlay_video", overlay_settings.output_video_path)
