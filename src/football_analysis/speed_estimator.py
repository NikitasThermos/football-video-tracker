from bbox_utils import measure_distance


def add_speed(players, total_frames, frame_window=5, frame_rate=24):
    for track_id in set(players["track_id"]):
        player = players[players["track_id"] == track_id]
        for frame_num in range(0, total_frames, frame_window):
            last_frame = min(frame_num + frame_window, total_frames - 1)

            start_position = player["position"][player["frame_num"] == frame_num]
            end_position = player["position"][player["frame_num"] == last_frame]

            if start_position.size == 0 or end_position.size == 0:
                continue

            distance = measure_distance(start_position[0], end_position[0])

            time_elapsed = (last_frame - frame_num) / frame_rate
            speed = distance / time_elapsed
            speed_km_per_hour = speed * 3.6

            players["speed"][
                (players["frame_num"] >= frame_num)
                & (players["frame_num"] <= last_frame)
                & players["track_id"]
                == track_id
            ] = speed_km_per_hour

    return players
