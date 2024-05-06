def compareFaceEncodings(unknown_encoding, known_encodings, known_names):
    duplicateName = ""
    distance = 0.0
    matches = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=0.5)
    face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)

    # Check if any face distance is not zero
    if np.any(face_distances < np.finfo(np.float64).max):
        best_match_index = np.argmin(face_distances)
        distance = face_distances[best_match_index]
        if matches[best_match_index]:
            acceptBool = True
            duplicateName = known_names[best_match_index]
        else:
            acceptBool = False
    else:
        acceptBool = False

    return acceptBool, duplicateName, distance
