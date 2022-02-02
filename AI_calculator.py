import cv2 as cv
import numpy as np
import argparse

from utils import*
from box import*

width = 150
height = 150

def main(args):
    cap = cv.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    hand = HandDetector(args.number_hand,args.model_complexity, args.min_detect, args.min_tracking)
    list_button = []
    imshow = ''

    # create number button
    button_zero = box(700, 150, width, height, str(0))
    list_button.append(button_zero)
    for i in range(3):
        for j in range(3):
            button = box(100 + j*200, 150 + i*200, width, height, str(i*3 + j + 1))
            list_button.append(button)

    # create sign button
    button_mul = box(700, 350, width, height, '*')
    list_button.append(button_mul)
    button_div = box(900, 350, width, height, '/')
    list_button.append(button_div)
    button_add = box(700, 550, width, height, '+')
    list_button.append(button_add)
    button_sub = box(900, 550, width, height, '-')
    list_button.append(button_sub)

    # create edit button
    button_DEL = box(900, 150, width, height, 'D')
    list_button.append(button_DEL)
    button_AC = box(1100, 150, width, height, 'A')
    list_button.append(button_AC)

    # create equal
    button_equal = box(1100, 350, 200, 350, '=')
    list_button.append(button_equal)

    # create result
    show_result = box(300, 20, 750, 100, '')


    while cap.isOpened():
        # time_since = time.time()
        timer = cv.getTickCount()
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv.flip(frame, 180)
        mark = np.zeros_like(frame)

        for button in list_button:
            button.render(mark)
        imshow = imshow[:12]
        show_result.name = imshow
        show_result.show_render(mark)

        # start hand tracking
        hand.findHand(frame)
        # create dict of landmark
        dict_cor = hand.dict_coordinate(frame)
        if len(dict_cor):
            dis, cx, cy = hand.calculate_distant(frame, dict_cor[8], dict_cor[12])
            for idx, button in enumerate(list_button):
                if button.x <= cx <= button.x + button.width and button.y <= cy <= button.y + button.height and dis <= 30:
                    button.is_torched = True
                    if button.one_torch:
                        if idx < 14:  # number and sign
                            imshow += button.name
                        if idx == 14:  # DEL
                            imshow = imshow[:-1]
                        if idx == 15:  # AC
                            imshow = ''
                        if idx == 16 and len(imshow):  # equal
                            try:
                                imshow = str(eval(imshow))
                            except:
                                imshow = 'Error'


                else:
                    button.is_torched = False

        fps = cv.getTickFrequency() / (cv.getTickCount() - timer)
        out = cv.addWeighted(frame, 0.6, mark, 0.4, 0)
        cv.putText(out, 'fps: {}'.format(str(int(fps))), (20, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv.imshow('img', out)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculator')
    parser.add_argument('--number-hand', type=int, default=1, help='number of hand detect')
    parser.add_argument('--model-complexity', type=int, default=0, help='choose between 0 and 1')
    parser.add_argument('--min-detect', type=float, default=0.5, help='min threadhold in detect module')
    parser.add_argument('--min-tracking', type=float, default=0.5, help='min threadhold in tracking module')
    args = parser.parse_args()

    main(args)

