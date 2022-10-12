"""QR Code decection."""
import cv2

cap = cv2.VideoCapture(1) # 0 is normal

qrCodeDetector = cv2.QRCodeDetector()

while True:
    ret, frame = cap.read()
    decodedText, points, _ = qrCodeDetector.detectAndDecode(frame)

    if points is not None:
        n_lines = len(points[0])
        bbox = points.astype(int)
        for i in range(n_lines):
            # draw all lines
            point1 = tuple(bbox[0, [i][0]])
            point2 = tuple(bbox[0, [(i+1) % n_lines][0]])
            cv2.line(frame, point1, point2, color=(255, 0, 0), thickness=2)

        print(decodedText)
    else:
        print("QR code not detected")

    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
