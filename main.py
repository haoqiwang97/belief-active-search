from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


app = FastAPI()
templates = Jinja2Templates(directory="templates")

# app.mount("/img_database_2d", StaticFiles(directory="./img_database_2d"), name="img_database_2d")

@app.get('/')
def index(request: Request):
    # return {"Hello": "World"}
    img1 = "https://www.w3schools.com/images/picture.jpg" #"/img_database_2d/13268_3D_165_6M_121311_UprightHH1_trim_clean_snapshot_noborder.png"
    img2 = "https://www.w3schools.com/images/w3schools_green.jpg" #"/img_database_2d/13589_3D_166v2_6M_110211_UprightHH1_trim_clean_snapshot_noborder.png"
    # read image to memory

    return templates.TemplateResponse("index.html", {"request": request, "img1": img1, "img2": img2})

"""
<html>
    <head>
        <title>show 2 images</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/semantic-ui@2.5.0/dist/semantic.min.css">
        <script src="https://cdn.jsdelivr.net/npm/semantic-ui@2.5.0/dist/semantic.min.js"></script>
    </head>
    <body>
        <div class="ui four column doubling stackable grid container">
            <div class="column">
                <img class="ui fluid image" src={{img1}}>
            </div>
            <div class="column">
                <img class="ui fluid image" src={{img2}}>
              <p></p>
            </div>
            <div class="column">
                <img class="ui fluid image" src={{img2}}>
              <p></p>
            </div>
            <div class="column">
              <p></p>
              <p></p>
            </div>
          </div>
          
        this is our content
        img1
        <img class="ui fluid image" src={{img1}}>
        img2
        <img class="ui fluid image" src={{img2}}>
        test image image
        <img class="ui small image" src="https://www.w3schools.com/images/picture.jpg">



    </body>
</html>
"""