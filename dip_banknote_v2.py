import cv2
import numpy as np
import os

def match_banknotes(main_image_path, template_dir):
    # Load the main image
    main_img = cv2.imread(main_image_path)
    main_gray = cv2.cvtColor(main_img, cv2.COLOR_BGR2GRAY)

    # Initialize the ORB detector
    orb = cv2.ORB_create()

    # Iterate over all templates in the directory
    for template_name in os.listdir(template_dir):
        template_path = os.path.join(template_dir, template_name)
        template_img = cv2.imread(template_path)
        template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(template_gray, None)
        kp2, des2 = orb.detectAndCompute(main_gray, None)

        # Use the Brute-Force Matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw matches for visualization (optional)
        result_img = cv2.drawMatches(template_img, kp1, main_img, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Find homography if enough matches are found
        if len(matches) > 10:  # Minimum threshold for robust matching
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Calculate Homography
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Use homography to locate the template in the main image
            h, w = template_gray.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            # Draw the bounding box
            main_img = cv2.polylines(main_img, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

        # Display results
        cv2.imshow(f'Matches for {template_name}', result_img)

    # Show the final image with bounding boxes
    cv2.imshow('Detected Banknotes', main_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function call
match_banknotes('./original.jpg', './templates/')
