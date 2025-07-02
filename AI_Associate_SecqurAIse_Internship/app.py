import cv2
import numpy as np
import time

def get_quadrant_number(x, y, width, height):
    """
    Returns the quadrant number (1-4) based on the position (x,y) of the ball in the frame.
    """
    if x < width/2 and y < height/2:
        return 1
    elif x >= width/2 and y < height/2:
        return 2
    elif x < width/2 and y >= height/2:
        return 3
    else:
        return 4
    
colors = {
    'blue': ([100, 50, 50], [130, 255, 255]),
    'red': ([0, 50, 50], [20, 255, 255]),
    'yellow': ([20, 100, 100], [30, 255, 255]),
    'white': ([0, 0, 200], [180, 20, 255])
}

cap = cv2.VideoCapture('ball_tracking.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
event_file = open('event_data.txt', 'w')

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for color, (lower, upper) in colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)

                if area > 100:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    timestamp = time.time()
                    quadrant = get_quadrant_number(x, y, width, height)
                    ball_color = color
                    event_type = "Entry"
                    event_file.write(f"{timestamp:.2f}, {quadrant}, {ball_color}, {event_type}\n")

                    quadrant = get_quadrant_number(x + w, y + h, width, height)
                    event_type = "Exit"
                    event_file.write(f"{timestamp:.2f}, {quadrant}, {ball_color}, {event_type}\n")

        out.write(frame)

        cv2.imshow('frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
event_file.close()

cv2.destroyAllWindows()

