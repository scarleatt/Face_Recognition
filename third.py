import matplotlib.pyplot as plt
import sys
import dlib
from skimage import io

face_detector = dlib.get_frontal_face_detector()

file_name=sys.argv[1]
dst=sys.argv[2]
image = io.imread(file_name)

detected_faces = face_detector(image, 0)
print("I found {} faces in the file {}".format(len(detected_faces), file_name))

for i, face_rect in enumerate(detected_faces):
    img = plt.imread(file_name)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    l=face_rect.left()
    r=face_rect.right()
    t=face_rect.top()
    b=face_rect.bottom()
    plt.vlines(l, b, t, colors="b", linewidth=2)
    plt.hlines(t, l, r, colors="b", linewidth=2)
    plt.hlines(b, l, r, colors="b", linewidth=2)
    plt.vlines(r, b, t, colors="b", linewidth=2)
    print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, l, t, r, b))

    plt.axis('off')
    fig.savefig('{}img_{}_rect_all_{}.jpg'.format(dst, file_name.split('.')[0],i),
                dpi=400, bbox_inches='tight')
    plt.close(fig)