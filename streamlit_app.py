if st.button("בדוק נוכחות"):
    if class_file is None:
        st.warning("יש להעלות תמונה")
        st.stop()

    class_image = Image.open(class_file)
    class_image = ImageOps.exif_transpose(class_image)
    
    faces, original_img_rgb = extract_faces(class_image, confidence)
    st.write(f"זוהו {len(faces)} פנים")

    present_students = {}
    recognized_faces = []

    for i, data in enumerate(faces):
        img = data["face"]
        box = data["box"]
        
        emb = model.predict(preprocess_image(img), verbose=0)[0]
        emb = emb / np.linalg.norm(emb)
        
        distances = []
        for name, ref_embs in reference_embeddings.items():
            for ref_emb in ref_embs:
                dist = cosine_distance(emb, ref_emb)
                distances.append((name, dist))
        
        distances.sort(key=lambda x: x[1])
        
        # *** DEBUG – הצג את המרחקים לכל פנים ***
        st.write(f"--- פנים #{i+1} ---")
        st.image(img, width=100, caption=f"פנים {i+1}")
        for name, dist in distances[:5]:
            st.write(f"{name}: {dist:.4f}")
        
        top_matches = distances[:5]
        votes = {}
        for name, dist in top_matches:
            if dist < threshold:
                votes[name] = votes.get(name, 0) + 1
        
        best_name = max(votes, key=votes.get) if votes else None
        st.write(f"זוהה כ: {best_name} (threshold={threshold})")
        
        if best_name and best_name not in present_students:
            present_students[best_name] = img
            recognized_faces.append({"name": best_name, "box": box})
