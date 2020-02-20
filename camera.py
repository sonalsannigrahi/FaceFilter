import cv2


def main():
    cap = cv2.VideoCapture(0)
    cascade = cv2.CascadeClassifier(
        "assets/haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()

        # Convert B&W
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blackwhite = cv2.equalizeHist(gray)

        rects = cascade.detectMultiScale(
            blackwhite, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)
            
        for x, y, w, h in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
