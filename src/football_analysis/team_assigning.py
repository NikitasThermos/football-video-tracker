import numpy as np
from sklearn.cluster import KMeans


def get_clustering_model(image):
    image_2d = image.reshape(-1, 3)

    kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
    kmeans.fit(image_2d)
    return kmeans


def get_player_color(frame, bbox):
    image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
    top_half_image = image[0 : int(image.shape[0] / 2), :]

    kmeans = get_clustering_model(top_half_image)
    labels = kmeans.labels_

    clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

    corner_clusters = [
        clustered_image[0, 0],
        clustered_image[0, -1],
        clustered_image[-1, 0],
        clustered_image[-1, -1],
    ]
    non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
    player_cluster = 1 - non_player_cluster
    player_color = kmeans.cluster_centers_[player_cluster]

    return player_color


def assign_team_color(frame, players):
    player_colors = []
    for player in players:
        bbox = player["bbox"]
        player_color = get_player_color(frame, bbox)
        player_colors.append(player_color)

    kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
    kmeans.fit(player_colors)

    team_colors = kmeans.cluster_centers_

    return team_colors, kmeans


def assign_teams(frames, players):
    team_colors, kmeans = assign_team_color(
        frames[0], players[players["frame_num"] == 0]
    )

    for player in players:

        if player["team"] == -1:
            bbox = player["bbox"]
            frame = frames[player["frame_num"]]
            player_color = get_player_color(frame, bbox)
            team_id = kmeans.predict(player_color.reshape(1, -1))[0]

            players["team"][players["track_id"] == player["track_id"]] = team_id
            players["color"][players["track_id"] == player["track_id"]] = np.array(
                team_colors[team_id]
            )

        if all(players["team"] != -1):
            break

    return players
