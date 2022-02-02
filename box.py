import cv2 as cv

class box():
    """ button object
    @parameter x, y is upper-left corner of button object
    @parameter width, height is shape of buttom object
    @parameter is_torched is changed when button is torched or not
    @parameter name is text on button
    @parameter one_torch is checked variance
    """
    def __init__(self, x, y, width, height, name):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.is_torched = False
        self.name = name
        self.one_torch = True

    def render(self, img):
        """ render the object on image
        Args:
            img: image object is drew on
        """
        if self.is_torched == True:
            self.one_torch = False
            cv.rectangle(img, (self.x, self.y), (self.x + self.width, self.y + self.height), (0, 255, 0), cv.FILLED)
        else:
            self.one_torch = True
            cv.rectangle(img, (self.x, self.y), (self.x + self.width, self.y + self.height), (255, 0, 255), cv.FILLED)

        cv.putText(img, self.name, (self.x + self.width//3, self.y + self.height*3//4),
                   cv.FONT_HERSHEY_COMPLEX, 3, (0, 0, 0), 3)

    def show_render(self, img):
        """ render the show-object on image
        Args:
            img: image object is drew on
        """
        cv.rectangle(img, (self.x, self.y), (self.x + self.width, self.y + self.height), (102, 102, 255), cv.FILLED)
        cv.putText(img, self.name, (self.x, self.y + self.height * 3 // 4),
                   cv.FONT_HERSHEY_COMPLEX, 3, (0, 0, 0), 3)