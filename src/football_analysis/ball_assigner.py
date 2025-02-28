from bbox_utils import get_center_of_bbox, measure_distance


def assign_ball_to_players(frames, players, ball, max_distance=70):
    team_ball_control = []
    for frame in range(len(frames)):
        players_in_frame = players[players["frame_num"] == frame]

        ball_bbox = ball["bbox"][ball["frame_num"] == frame]
        ball_position = get_center_of_bbox(*ball_bbox)

        closest_player = None
        min_distance = float("inf")

        for player in players_in_frame:
            player_bbox = player["bbox"]

            distance_left = measure_distance(
                (player_bbox[0], player_bbox[-1]), ball_position
            )
            distance_right = measure_distance(
                (player_bbox[2], player_bbox[-1]), ball_position
            )
            distance = max(distance_left, distance_right)

            if distance < min_distance and distance < max_distance:
                min_distance = distance
                closest_player = player
        if closest_player:
            players["has_ball"][
                (players["track_id"] == closest_player["track_id"])
                & (players["frame_num"] == frame)
            ] = True
            team_ball_control.append(closest_player["team"])
        else:
            if not team_ball_control:
                team_ball_control.append(-1)
            else:
                team_ball_control.append(team_ball_control[-1])

    return players, team_ball_control
