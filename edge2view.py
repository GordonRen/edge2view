import argparse
import cv2
import numpy as np
import tensorflow as tf

CROP_SIZE = 256
drawing = False
px, py = -1, -1
draw_img = np.zeros((256, 256, 3), np.uint8) + 255

def load_graph(frozen_graph_filename):
    """Load a (frozen) Tensorflow model into memory."""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_filename, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph

def draw_circle(event, x, y, flags, param):
    global px, py, px, py, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        print 'EVENT_LBUTTONDOWN'
        drawing = True
        px, py = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            print 'EVENT_MOUSEMOVE'
            cv2.circle(draw_img, (x, y), 1, (0, 0, 0), -1)
            cv2.line(draw_img, (px, py), (x, y), (0, 0, 0), 2)
            px, py = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        print 'EVENT_LBUTTONUP'
        cv2.circle(draw_img, (x, y), 1, (0, 0, 0), -1)

def main():
    #TensorFlow
    graph = load_graph(args.frozen_model_file)
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    output_tensor = graph.get_tensor_by_name('generate_output/output:0')
    sess = tf.Session(graph=graph)
    generated_image = np.zeros((256, 256, 3), np.uint8) + 255
    cv2.namedWindow('edge2view')
    cv2.setMouseCallback('edge2view', draw_circle)
    while True:
        # generate prediction
        combined_image = np.concatenate([draw_img, generated_image], axis=1)
        image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR instead of RGB
        output_image = sess.run(output_tensor, feed_dict={image_tensor: image_rgb})
        image_bgr = cv2.cvtColor(np.squeeze(output_image), cv2.COLOR_RGB2BGR)
        image_normal = np.concatenate([draw_img, image_bgr], axis=1)

        cv2.imshow('edge2view', image_normal)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    sess.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf-model', dest='frozen_model_file', type=str, default='edge2view-reduced-model/frozen_model.pb',help='Frozen TensorFlow model file.')
    args = parser.parse_args()
    main()