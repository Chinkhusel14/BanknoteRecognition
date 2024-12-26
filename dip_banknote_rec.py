import cv2
import os

def preprocess_image(image):
    # Convert an image to grayscale for easier processing
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def segment_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segments = []
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box with thickness 2
            
            segments.append((x, y, w, h))

    cv2.imshow('Segmented Image with Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return segments


def resize_to_same_dimensions(image1, image2):
    height = min(image1.shape[0], image2.shape[0])
    width = min(image1.shape[1], image2.shape[1])
    resized_image1 = cv2.resize(image1, (width, height))
    resized_image2 = cv2.resize(image2, (width, height))

    # Calculate scale factors to map back to original dimensions
    return resized_image1, resized_image2, (image1.shape[1] / width, image1.shape[0] / height)

def match_banknotes(main_image_path, template_dir):
    main_img = cv2.imread(main_image_path)

    segments = segment_image(main_img)

    for template_name in os.listdir(template_dir):
        # Load each template image
        template_path = os.path.join(template_dir, template_name)
        template_img = cv2.imread(template_path)
        template_gray = preprocess_image(template_img)

        for x, y, w, h in segments:
            segment_img = main_img[y:y+h, x:x+w]
            segment_gray = preprocess_image(segment_img)

            segment_resized, template_resized, scale_factors = resize_to_same_dimensions(segment_gray, template_gray)

            # Perform template matching
            result = cv2.matchTemplate(segment_resized, template_resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            print(f"Template: {template_name}, Match value: {max_val}, Location: {max_loc}")

            if max_val > 0.5:  # Match threshold
                # Track the best match for each segment
                if 'best_match' not in locals() or max_val > best_match['value']:
                    best_match = {
                        'value': max_val,
                        'location': max_loc,
                        'template_name': template_name,
                        'segment': (x, y, w, h),
                        'scale_factors': scale_factors
                    }

        # After checking all templates for a segment, draw the best match
        if 'best_match' in locals():
            x, y, w, h = best_match['segment']
            max_loc = best_match['location']
            template_name = best_match['template_name']
            scale_factors = best_match['scale_factors']

            # Display only the name near the segmented region
            text_x = x + 10
            text_y = y + h + 20
            cv2.putText(main_img, template_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Reset best match for next segment
            del best_match

    # Display the final annotated image
    cv2.imshow('Final Detection (Template Matching)', main_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function call
match_banknotes('./original.jpg', './templates/')
