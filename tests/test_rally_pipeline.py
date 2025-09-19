import unittest

from decision.rally_processor import RallyProcessor
from decision.state_machine import MatchStateMachine


def _make_clip(cid: str, cls: str, start: int, end: int, team: str) -> dict:
    return {
        "id": cid,
        "class": cls,
        "start": start,
        "end": end,
        "team_name": team,
        "mean_conf": 0.9,
        "peak_conf": 0.95,
    }


class RallyPipelineTests(unittest.TestCase):
    def test_rally_processor_detects_ace_and_scores(self) -> None:
        teamA = "TeamA"
        teamB = "TeamB"
        clips = [
            _make_clip("serve1", "serve", 0, 3, teamA),
        ]
        court = {
            0: [(0.0, 0.0), (1800.0, 0.0), (1800.0, 900.0), (0.0, 900.0)],
            5: [(0.0, 0.0), (1800.0, 0.0), (1800.0, 900.0), (0.0, 900.0)],
        }
        ball_tracks = {
            2: {"x": 200.0, "y": 200.0, "confidence": 0.9},
            6: {"x": 1600.0, "y": 500.0, "confidence": 0.92},
        }
        side_to_team = {"left": teamA, "right": teamB}

        rp = RallyProcessor(
            clips=clips,
            ball_tracks=ball_tracks,
            court_timeseries=court,
            fps=30.0,
            teamA=teamA,
            teamB=teamB,
            side_to_team=side_to_team,
            players_by_frame=None,
            mapper_dims=(1800, 900),
        )

        rallies = rp.rallies()
        self.assertEqual(len(rallies), 1)
        rally = rallies[0]
        self.assertIsNotNone(rally.serve)
        self.assertEqual(rally.serve_result, "ace")
        self.assertEqual(rally.winner_team, teamA)
        self.assertIn(rally.end_reason, {"ground", "ace"})
        self.assertEqual(rally.decisive_frame, 6)
        self.assertTrue(rally.ball_events)
        self.assertEqual(rally.ball_events[-1].kind, "ground")

        msm = MatchStateMachine(
            teamA=teamA,
            teamB=teamB,
            score_flash_frames=6,
            min_action_conf=0.0,
        )

        score_lines = []
        lines_at6 = None
        for fi in range(0, 12):
            ctx = rp.context_for_frame(fi)
            lines = msm.process(ctx)
            score_lines.append(lines[0])
            if fi == 6:
                lines_at6 = list(lines)
        self.assertEqual(score_lines[0], "TeamA 0 - 0 TeamB")
        self.assertEqual(score_lines[-1], "TeamA 1 - 0 TeamB")

        final_lines = msm.process(rp.context_for_frame(12))
        self.assertTrue(any("TeamA +1" in line for line in final_lines))
        self.assertIsNotNone(lines_at6)
        self.assertTrue(any("TeamA serve" in line for line in lines_at6))
        self.assertTrue(any("Ball ground" in line for line in lines_at6))


if __name__ == "__main__":
    unittest.main()
