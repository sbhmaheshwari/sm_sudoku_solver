from flask import *
import cv2
from get_sudoku_grid import *
from solve_sudoku import *
import imutils
from imutils.video import VideoStream
import time
import threading
import tensorflow as tf

outputFrame = None
lock = threading.Lock()

app = Flask(__name__)

model_ch4 = tf.keras.models.load_model('model_ch4.hdf5')
vs = cv2.VideoCapture(0)
time.sleep(2.0)


@app.route('/')
def index():
    return(render_template('index.html'))


def put_text(warp_img, num, coords, font = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (255, 0, 0), thickness = 2):
    '''put text on the image'''
    img = warp_img.copy()
    x, y = coords
    x, y = int(x), int(y)
    m_img = cv2.putText(img, str(num), (x,y), font, fontScale, color, thickness)
    return(m_img)

# this function runs in background and updates outputFrame 
def get_solved_sudoku():
    global vs, outputFrame, lock
    val_init = {}
    while True:
        _, frame = vs.read()
        img_r = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed_img = preprocess_image(img_r)
        req_points = get_corners(processed_img)
        warped_image = get_warped(img_r, req_points)
        squares = infer_grid(warped_image)
        try:
            sud_digits = get_sudoku(img = frame)
            sum_digits = sum([np.sum(i) for i in sud_digits])
            if (sum_digits)>0:
                digits = [i[np.newaxis,:,:,np.newaxis]/255 for i in sud_digits]
                grid_preds = [str(model_ch4.predict(i).argmax(-1)[0]+1)if np.sum(i)>0 else '.' for i in digits]
                #print(grid_preds)
                valid_sum = sum([False if i in '0.' else True for i in grid_preds])
        
                if valid_sum>15:
                    val = solve(grid_preds)
            
                    if val is not False:
                        val_init = val
                        empty = '0.'
                        fin_img = warped_image.copy()
                        #print(fin_img)
                        for ind, i in enumerate(grid_preds):
                            if i in empty: 
                                x1 = (squares[ind][0][0]+squares[ind][1][0])/2
                                x2 = (squares[ind][0][1]+squares[ind][1][1])/2
                                fin_img = put_text(fin_img, val[list(val)[ind]], 
                                             ((x1+squares[ind][0][0])/2, (x2+squares[ind][1][1])/2))
                        
                        with lock:
                            print('solved!')
                            outputFrame = fin_img.copy()
        except:
            continue

# Decorator for displaying sudoku image        
def generate():
    global lock, outputFrame
    while True:
        with lock:
            if outputFrame is None: continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag: continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

# Decorator for displaying real time camera image            
def generate2():
    global lock
    while True:
        _, frame_out = vs.read()
        if frame_out is None: continue
        (flag, encodedImage2) = cv2.imencode(".jpg", frame_out)
        if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage2) + b'\r\n')



@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(generate2(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # defining server ip address and port
    t = threading.Thread(target = get_solved_sudoku)
    t.daemon = True
    t.start()
    print('started')
    app.run(host='localhost', port=9000, debug=True,
		threaded=True, use_reloader=False)

vs.release()
cv2.destroyAllWindows()