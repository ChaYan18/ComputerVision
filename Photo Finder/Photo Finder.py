import cv2, sys, numpy, os, shutil, glob, time, tkMessageBox, tkFileDialog, ttk
from Tkinter import *
from PIL import Image, ImageTk
from tkFileDialog import askopenfilename
from os import listdir
from os.path import isfile, join

haar_file = 'cascade_rai.xml'
datasets = 'datasets'
sub_data = 'Ryan'
Album = 'album_source'
RESIZE_FACTOR = 4
global dirpath
global reference
global bar

#checking the path of datasets
path1 = os.path.join(datasets, sub_data)
if not os.path.isdir(path1):
    os.mkdir(path1)
    
#Loading the HaarCascade file 
face_cascade = cv2.CascadeClassifier(haar_file)

def description():
    descrip=tkMessageBox.showinfo(
        message='Photo Finder : Face Recognition Application%s'
        %'\n\nImplementation:%s' %'\n\nEigenFace Algorithm\nLBPH Algorithm%s'
        %'\nPython 2.7\nOpenCV 2.4.13\nPIL 3.0\nTkinter Gui')

def about():
    about = tkMessageBox.showinfo(
        title="About - Photo Finder",
        message = 'Photo Finder - Face Recognition Application%s'
        %'\n\nBSCpE 4-2\nBanagan Ellen\nIraola Ryan\nMalaza Camille\nQuebada Charlene%s'
        %'\n\n------Sept-2016-------')
 
def g_quit(): #for exit button under menu
    mExit=tkMessageBox.askyesno(title="Exit", message="Confirm Exit ?")
    if mExit>0:
        mGui.destroy()
        
def browseFolder(): #browse to select the directory of the album of photos
    global dirpath
    dirname = tkFileDialog.askdirectory(parent=ActionFrame, title='Please select Album folder')
    dirpath = '%s'%dirname
    return dirpath

def finished(): # close app 
    done = tkMessageBox.askokcancel(title="DONE SORTING", message="DONE SORTING !  Press Ok to Exit")
    if done>0:
        mGui.destroy()

def new(): #clear the canvas when pressed new in menu
    
    label2.pack_forget()
    textlabel2.pack_forget()
    textlabel2.configure(text='Please Select Reference Photo',width=50,wraplength=140,fg="BLUE")
    textlabel2.pack()
        
#open menu
def open_img():  #grab selected image and display in canvas
    global reference
    file = askopenfilename(initialdir='C:\Python27\FaceRecog_Rye1') #file - string name of selected reference photo
    w_box = 400
    h_box = 400
    pil_image = Image.open(file)
    textlabel2.config(text=file,width=50,wraplength=500)
    w, h = pil_image.size
    pil_image_resized = resize(w, h, w_box, h_box, pil_image)
    reference =file
    tk_image = ImageTk.PhotoImage(pil_image_resized)
    label2.tk_image = ImageTk.PhotoImage(pil_image_resized)
    label2.config(image=label2.tk_image, width=w_box, height=h_box)

    label2.pack()
    return reference

def resize(w, h, w_box, h_box, pil_image): 
    '''
    resize a pil_image object so it will fit into
    a box of size w_box times h_box, but retain aspect ratio
    '''
    f1 = 1.0*w_box/w  # 1.0 forces float division in Python2
    f2 = 1.0*h_box/h
    factor = min([f1, f2])
    #print(f1, f2, factor)  # test
    # use best down-sizing filter
    width = int(w*factor)
    height = int(h*factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)

def start():#Face recognition and sorting

    
    
    print ' \t \t \t \t " Photo Finder " \n \n'
    namecount = 0 #naming photos to be sorted
    foldername = svalue.get()
    os.mkdir(foldername) #make the directory of 
    
    #Path of Cropped Face Datas
    #Creating Array of Images from the Album
    onlyfiles = [f for f in listdir(dirpath) if isfile(join(dirpath,f))]
    images1 = numpy.empty(len(onlyfiles),dtype=object)

    #set width and height of cropped faces
    (width, height) = (112, 92)

    progressbar.start()
    for n in range(0, len(onlyfiles)):
        count = 0 # count :used for naming the photos in datasets
        #images[n] --> create array of images in Album
        images1[n] = cv2.imread( join(dirpath,onlyfiles[n]))
        
        gray1 = cv2.cvtColor(images1[n], cv2.COLOR_BGR2GRAY)
        gray_resized0 = cv2.resize(gray1, (gray1.shape[1]/RESIZE_FACTOR, gray1.shape[0]/RESIZE_FACTOR)) 
        faces1 = face_cascade.detectMultiScale(
                           gray_resized0,
                           scaleFactor=1.2,
                           minNeighbors=5,
                           minSize =(40,40),
                           flags=cv2.cv.CV_HAAR_SCALE_IMAGE
                               )
        #Apply face detection for every numpy array images
        #crop the faces with given width and height
        for (x,y,w,h) in faces1:
            
            face1 = gray_resized0[y:y + h, x:x + w]
            face_resize1 = cv2.resize(face1, (width, height))
            #save the cropped gray face to path1
            cv2.imwrite('%s/%s.png' % (path1,[count]), face_resize1)
                
            '''CREATING THE MODEL'''

            # Create a list of images and a list of corresponding names
            (images, lables, names, id) = ([], [], {}, 0)

            for (subdirs, dirs, files) in os.walk(datasets):
                for subdir in dirs:
                    names[id] = subdir
                    subjectpath = os.path.join(datasets, subdir)
                    for filename in os.listdir(subjectpath):
                                path = subjectpath + '/' + filename
                                lable = id
                                images.append(cv2.imread(path, 0))
                                lables.append(int(lable))
                                id += 1
                #create numpy array of list created
                (images, lables) = [numpy.array(lis) for lis in [images, lables]]

            #Load the LBPH FaceRecognizer method and Eigen FaceRecognizer
            recognizer = cv2.createLBPHFaceRecognizer()
            eigen = cv2.createEigenFaceRecognizer()
            #OpenCV trains a model from the images
            recognizer.train(images, lables)
            eigen.train(images,lables)
                
            #Reading the frame capture by webcam after the Space key pressed
            print "reading . . ."
            progressbar.step(3) #adding steps to progressbar
            user = cv2.imread("%s"%reference)# read the reference image from gui
            
            '''TEST DATA'''

            #Apply Face detection to reference photo
            
            gray = cv2.cvtColor(user, cv2.COLOR_BGR2GRAY)
            gray_resized = cv2.resize(gray, (gray.shape[1]/RESIZE_FACTOR, gray.shape[0]/RESIZE_FACTOR)) 
            faces = face_cascade.detectMultiScale(
                           gray_resized,
                           scaleFactor=1.3,
                           minNeighbors=5,
                           minSize=(30,30),
                           flags=cv2.cv.CV_HAAR_SCALE_IMAGE
                           )
            #crop face from reference photo
            for (x,y,w,h) in faces:
                face = gray_resized[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (width, height))
                    
            # Try to recognize the cropped face
            #Prediction = mismatch percentage

                prediction2 = recognizer.predict(face_resize)
                prediction1 = eigen.predict(face_resize)
                print prediction2[1]
                print prediction1[1]
                        
                if (prediction2[1]<=100 and prediction1[1] < 2000) or (prediction2[1]<=150 and prediction1[1] > 2210 and prediction1[1] < 2600):
                    print "   0000  Result:   Image Found !"
                    print prediction2[1]
                    cv2.imwrite('%s/%s.jpg' % ("C:\Python27\FaceRecog_Rye1\%s"%foldername,namecount), images1[n])
                    namecount += 1

                if prediction2[1]<=117 and prediction2[1] >= 100 and prediction1[1] > 1000 and prediction1[1] < 2600 or prediction2[1] >= 118 and prediction2[1] <=120  and prediction1[1] > 1000 and prediction1[1] < 2600:
                    print "     Result:   Image Found !"
                    print prediction2[1]
                    cv2.imwrite('%s/%s.jpg' % ("C:\Python27\FaceRecog_Rye1\%s"%foldername,namecount), images1[n])
                    namecount += 1
                    
                

                '''src = "C:\\Python27\\FaceRecog_Rye1\\album_source"
                dst = foldername

                for jpgfile in glob.iglob(os.path.join(src,"*.jpg")):
                    shutil.move(jpgfile, dst)
                    key = cv2.waitKey(1) '''     
                count += 1
            progressbar.update_idletasks() #update the progressbar    
            break
    
    progressbar.stop()        
    print "     FINISHED SORTING"  #exit when done
    finished()
 

mGui = Tk()
mGui.title('Photo Finder')
mGui.geometry('550x400')
mGui.resizable(1, 1) #Disable Resizeability of the window
photoFrame = Frame(mGui, bg="gray")
photoFrame.pack(side=LEFT)
ActionFrame = Frame(mGui, bg="black", width=100, height=200)
ActionFrame.pack(side= RIGHT, fill=X and Y)
textlabel2 = Label(photoFrame,text='Please Select Reference Photo',width=50,wraplength=140,fg="BLUE")
textlabel2.pack()
label2 = Label(photoFrame)

#just color spacing
space=Label(ActionFrame, bg="BLACK")
space.pack()
space=Label(ActionFrame, bg="BLACK")
space.pack()
space=Label(ActionFrame, bg="BLACK")
space.pack()

#Create Buttons and actions
textmsg = Label(ActionFrame, text="Username", bg="BLACK", fg = "white")
textmsg.pack()
svalue = StringVar()
w = Entry(ActionFrame,textvariable=svalue) # adds a textarea widget that accepts the username
w.pack()
space=Label(ActionFrame, bg="BLACK")
space.pack()
browse_btn = Button(ActionFrame, text= "       Select Album Folder \t.", command=browseFolder)
browse_btn.pack()
space=Label(ActionFrame, bg="BLACK")
space.pack()
negative_btn = Button(ActionFrame, text="        Start Photo Finder \t.", command=start)
negative_btn.pack()

# ProgressBar
mainframe = ttk.Frame(ActionFrame)
mainframe.grid(column=2, row=2, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)
mainframe.pack(side= RIGHT, fill=X)
progressbar = ttk.Progressbar(mainframe,length =200, mode='determinate')
progressbar.grid(column=1, row=100, sticky=W)

#Menu Bar
menubar = Menu(mGui)
filemenu = Menu(menubar)
helpm = Menu(mGui)

#Create the Menu Options that go under drop down
filemenu.add_command(label="Open Reference Photo", command=open_img)
filemenu.add_command(label="New", command=  new)
filemenu.add_command(label="Exit", command=g_quit)
helpm.add_command(label="About", command=about)
helpm.add_command(label="Description", command=description)


#Create the Main Button (e.g file) which contains the drop down options
menubar.add_cascade(label="Menu", menu=filemenu)
menubar.add_cascade(label="Help", menu=helpm)

mGui.config(menu=menubar)
mGui.mainloop()
