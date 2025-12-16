import cv2
import numpy as np

class CVButtonDetector:
    def __init__(self):
        pass

    def get_robust_red_mask(self,hsv_img):
        # Lower Red (0-10)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        
        # Upper Red (170-180)
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
        
        # Combine
        mask = mask1 + mask2
        return mask
    
    def detect_red_round_button(self,img_bgr):
        output = img_bgr.copy()
        mask = self.get_robust_red_mask(cv2.cvtColor(cv2.GaussianBlur(img_bgr, (5,5), 0), cv2.COLOR_BGR2HSV))

        # 1. Preprocess for Hough Circles (Grayscale + Blur)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)*mask
        # Blur is crucial to avoid false circles
        gray_blurred = cv2.medianBlur(gray, 5) 

        # 2. Detect Circles
        # param1: Canny edge threshold (high)
        # param2: Accumulator threshold (lower = more circles detected)
        # minRadius/maxRadius: Adapt these to your button size in pixels!
        circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                param1=100, param2=30, minRadius=30, maxRadius=1000)

        best_circle = None
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # 3. Check Color at the center of the circle
                # Create a small mask around the center to average the color
                mask = np.zeros(img_bgr.shape[:2], dtype="uint8")
                cv2.circle(mask, (x, y), int(r/2), 255, -1)
                
                # Get average color in HSV
                mean_val = cv2.mean(img_bgr, mask=mask) # Returns BGR mean
                mean_bgr = np.array([[[mean_val[0], mean_val[1], mean_val[2]]]], dtype=np.uint8)
                mean_hsv = cv2.cvtColor(mean_bgr, cv2.COLOR_BGR2HSV)[0][0]
                
                h_val = mean_hsv[0]
                s_val = mean_hsv[1]
                v_val = mean_hsv[2]

                # 4. Wide Red Filter
                # Red wraps around 180. It is usually 0-10 AND 170-180.
                # We also ensure Saturation is high enough (it's not white/grey)
                is_red = ((h_val < 10) or (h_val > 160)) and (s_val > 50) and (v_val > 50)

                if is_red:
                    best_circle = (x, y)
                    # Draw for visualization
                    cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                    break # Stop after finding the first one, or add logic to find the "best" one

        if best_circle is None:
            return None, None, None, output

        # Calculate errors
        cx, cy = best_circle
        tx, ty = (0,0) #TODO remove this
        error_x = cx - tx
        error_y = cy - ty
        dist = np.sqrt(error_x**2 + error_y**2)

        return best_circle, (error_x, error_y), dist, output