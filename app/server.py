from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
import numpy as np
from fastai import *
from fastai.vision import *

# export_file_url = 'https://www.dropbox.com/s/v6cuuvddq73d1e0/export.pkl?raw=1'
#'https://drive.google.com/uc?export=download&id=19P3D8dtv6GnlIbgDOwaFQplgzy2fC79_'
export_file_url ='https://drive.google.com/uc?export=download&id=1Nu1G_gpSSOL0VJsklqYDlXroLP0PZXyX'
export_file_name = 'export.pkl'

classes = ['good_grain','bad_grain']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)
         
def pca(data):
        dvect = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data[i, j]: dvect.append([i, j])
        dvect = np.array(dvect, dtype=np.float32)
        dvect = np.array(dvect) - np.mean(dvect, axis=0)
        dvect /= np.std(dvect, 0)
        cov = np.dot(dvect.T, dvect) / dvect.shape[0]
        eigenval, eigenvect = np.linalg.eigh(cov)
        return cov, eigenvect, eigenval   

def get_files(indir):
    indir = indir.rstrip('/')
    flist =os.listdir(indir)
    files = []
    for f in flist:
        f = indir+'/'+f
        if os.path.isdir(f):
            tfiles = get_files(f)
            files += [tf for tf in tfiles]
        else:
            files.append(f)
    return files
           
async def setup_learner():
    await download_file(export_file_url, path/export_file_name)
    try:
        img_new = cv2.imread(str(path),cv2.IMREAD_COLOR)
        img_new = cv2.fastNlMeansDenoisingColored(img_new,None,10,10,7,21)
        img_sv=img_new
        img_test=img_sv
        img=img_new
        r = img.copy()
        r[:, :, 0] = 0
        r[:, :, 1] = 0
        img=r
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        test_r=img
        im_gray = cv2.cvtColor(r,cv2.COLOR_BGR2GRAY)
        im_blur=cv2.GaussianBlur(im_gray,(5,5),0)
        kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                            [-1,-1,-1]])
        im_sharp = cv2.filter2D(im_blur, -1, kernel_sharpening)
        img=im_sharp
        th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        img=im_sharp
        ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh3,cv2.MORPH_OPEN,kernel, iterations = 2)
        ret,th = cv2.threshold(thresh3,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(th,cv2.MORPH_OPEN,kernel, iterations = 2)
        sure_bg = cv2.dilate(th,kernel,iterations=3)
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.005*dist_transform.max(),255,0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers+1
        markers[unknown==255] = 0
        markers = cv2.watershed(img_sv,markers)
        img_sv[markers == -1] = [0,255,0]
        img_sv[markers == 1] = [0,0,0]
        from skimage.filters import threshold_otsu
        thresh_val = threshold_otsu(opening)
        mask = np.where(opening > thresh_val, 1, 0)
        if np.sum(mask==0) < np.sum(mask==1):
            mask = np.where(mask, 0, 1)
        from scipy import ndimage
        labels, nlabels = ndimage.label(mask)
        all_cords={}
        for label_ind, label_coords in enumerate(ndimage.find_objects(labels)):
            cell = im_gray[label_coords]
            all_cords[label_ind]=label_coords
            if np.product(cell.shape) < 10: 
                mask = np.where(labels==label_ind+1, 0, mask)
        labels, nlabels = ndimage.label(mask)
        i=0
        res={}
        for ii, obj_indices in enumerate(ndimage.find_objects(labels)[0:nlabels]):
            cell = img_sv[obj_indices]
            res[i]=cell
            i=i+1
        i=0
        features = {}
        for i in res:
            gcolor = res[i]
            h, w, _ = gcolor.shape
            ggray = np.array([[gcolor[i,j,2] for j in range(w)] for i in range(h)], dtype=np.uint8)
            area = np.sum(np.sum([[1.0 for j in range(w) if ggray[i, j]] for i in range(h)]))
            mean_area = area / (h * w)
            r, b, g = np.sum([gcolor[i, j] for j in range(w) for i in range(h)], axis=0) / (area * 256)
            _, _, eigen_value = pca(ggray)
           eccentricity = eigen_value[0] / eigen_value[1]
           l = [mean_area, r, b, g, eigen_value[0], eigen_value[1], eccentricity]
           features[i] = np.array(l)
           i=i+1
       
        out={}
        #learn = load_learner(path, export_file_name)
        #Change Working directory of pkl file
        model = keras.models.load_model('../input/weight/weights.pkl')
        out = {}
        for i in features:
            out[i] = model.predict(np.array([features[i]]))
        good = not_good = 0
        for i in out:
            s = res[i]
            if np.argmax(out[i][0]) == 0:
                good += 1
                x1=all_cords[i][0].start
                y1=all_cords[i][1].start
                x2=all_cords[i][1].stop
                y2=all_cords[i][0].stop
                cv2.rectangle(img_test,(x2,x1),(y1,y2), (255, 0, 0),8)
            else:
                x1=all_cords[i][0].start
                y1=all_cords[i][1].start
                x2=all_cords[i][1].stop
                y2=all_cords[i][0].stop
                not_good+=1
                cv2.rectangle(img_test,(x2,x1),(y1,y2), (0, 0, 255), 3)
        p=(good/(good+not_good)*100)
        print("Number of good grain :", good)
        print("Number of impure grains or impurity:", not_good)
        print("Percentage Purity is:",p)
        return p
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
