import cv2 as cv
import mediapipe as mp
import time
import math

class HandDetector():
    """custom from mediapipe hand tracking
    @parameter max_num_hands is number of hands detect
    @parameter model_complexity raise speech or raise accuracy
    @parameter min_detection_confidence min threadhold in module detect
    @parameter min_tracking_confidence min threadhold in module tracking
    """
    def __init__(self, max_num_hands=2, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            max_num_hands=self.max_num_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

    def findHand(self, frame):
        """
        Args:
            frame: is image drew result of module
        """
        imgRGB = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw_styles.get_default_hand_landmarks_style(),
                    self.mp_draw_styles.get_default_hand_connections_style())

    def dict_coordinate(self, frame):
        """
        Args:
            frame: the image which is got landmark feature
        Return:
            A dict of landmark feature (number of landmark: (id of hand, x, y))
        """
        dict_cor = {}
        h, w, c = frame.shape
        if self.results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
                for id, lm in enumerate(hand_landmarks.landmark):
                    dict_cor[id] = idx, int(lm.x * w), int(lm.y * h)
        return dict_cor

    def calculate_distant(self, frame, pt1, pt2, draw=True):
        """
        Args:
            frame: the image which is used for calculate
            pt1: is a point landmark
            pt2: is a point landmark
            draw: whether draw or not
        Return:
            distance from pt1 to pt2, coordinate of center between pt1 and pt2
        """
        x_dis = abs(pt1[1]-pt2[1])
        y_dis = abs(pt1[2]-pt2[2])
        cx = min(pt1[1], pt2[1]) + x_dis//2
        cy = min(pt1[2], pt2[2]) + y_dis//2
        distance = math.sqrt(x_dis*x_dis + y_dis*y_dis)
        if draw:
            cv.line(frame, (pt1[1], pt1[2]), (pt2[1], pt2[2]), (0, 255, 0), 3)
            cv.circle(frame, (cx, cy), 5, (255, 0, 255), cv.FILLED)
        return distance, cx, cy

if __name__ == '__main__':
    hand = HandDetector(max_num_hands=1)
    cap = cv.VideoCapture(0)
    while cap.isOpened():
        time_since = time.time()
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv.flip(frame, 1)
        hand.findHand(frame)
        dict_cor = hand.dict_coordinate(frame)
        if len(dict_cor):
            print(hand.caculatate_distant(frame, dict_cor[8], dict_cor[12]))
        else:
            print('no hand detected')
        time_after = time.time()
        fps = 1/(time_after - time_since)

        cv.putText(frame, str(int(fps)), (20, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv.imshow('Hand_detect', frame)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

