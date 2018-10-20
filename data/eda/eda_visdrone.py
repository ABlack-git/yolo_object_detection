import cv2
import os
import stats_utils as su


def compute_aspectratio(width: int, height: int):
    def gcd(a, b):
        return a if b == 0 else gcd(b, a % b)

    if width == height:
        return '1:1'
    r = gcd(width, height)
    x = int(width / r)
    y = int(height / r)
    return '{0}:{1}'.format(x, y)


def as_stats():
    img_dir = '/Volumes/TRANSCEND/Data Sets/VisDrone2018-DET-train/images'
    imgs = [f for f in os.listdir(img_dir) if (not f.startswith('.')) and f.endswith('.jpg')]
    no_items = len(imgs)
    aspects = {}
    wh = {}
    for i, img_path in enumerate(imgs):
        img = cv2.imread(os.path.join(img_dir, img_path))
        asr = compute_aspectratio(img.shape[1], img.shape[0])
        dims = '{0}:{1}'.format(img.shape[1], img.shape[0])
        su.progress_bar(i + 1, no_items)
        if asr not in aspects:
            aspects[asr] = 0
        if dims not in wh:
            wh[dims] = 0
        aspects[asr] += 1
        wh[dims] += 1
    print(aspects)
    print(wh)


if __name__ == '__main__':
    as_stats()
