# Converts the raw image to numpy array , 
# Approproiate input dimensions for fittign into Vggnet architecture 

def preprocess_image_from_url(image_url):
    fd = urllib.urlopen(image_url)
    image_file = io.BytesIO(fd.read())
    img = Image.open(image_file)
    #img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224), Image.ANTIALIAS)  # resize the image to fit into VGG-16
    img = np.array(img.getdata(), np.uint8)
    img = img.reshape(224, 224, 3).astype(np.float32)
    img[:,:,0] -= 123.68 # subtract mean (probably unnecessary for t-SNE but good practice)
    img[:,:,1] -= 116.779
    img[:,:,2] -= 103.939
    img = img.transpose((2,0,1))
    img = np.expand_dims(img, axis=0)
    return img