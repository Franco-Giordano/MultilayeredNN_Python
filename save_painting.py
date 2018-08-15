from PIL import ImageDraw
from tkinter import *
import PIL

width = 200
height = 200
black = (0, 0, 0)

class Painter:
    def __init__(self, width, height, bg_colour):
        
        self.width = width
        self.height = height
        self.bg_colour = bg_colour
        self.image1 = 0
        self.root = Tk()
        self.root.resizable(0, 0)
        self.cv = Canvas(self.root, width=width, height=height, bg='white')
        self.draw = 0


    def save(self):
        filename = "image.png"
        self.image1.thumbnail((28,28), PIL.Image.ANTIALIAS)
        self.image1.save(filename)
        self.root.destroy()
    
    def paint(self, event):
        x1, y1 = (event.x - 3), (event.y - 3)
        x2, y2 = (event.x + 3), (event.y + 3)
        self.cv.create_oval(x1, y1, x2, y2, fill="white",width=5)
        self.draw.line([x1, y1, x2, y2],fill="white",width=5)
        
    def reset(self):
        self.cv.delete("all")
        self.image1 = PIL.Image.new("RGB", (width, height), black)
        self.draw = ImageDraw.Draw(self.image1)
            
    def create_canvas(self):
        
        # Tkinter create a canvas to draw on
        self.cv.pack()
        
        # PIL create an empty image and draw object to draw on
        # memory only, not visible
        self.image1 = PIL.Image.new("RGB", (width, height), black)
        self.draw = ImageDraw.Draw(self.image1)
        
        self.cv.pack(expand=NO, fill=NONE)
        self.cv.bind("<B1-Motion>", self.paint)
        
        button=Button(text="save",command=self.save)
        button.pack()
        
        reset = Button(text = "reset", command= self.reset)
        reset.pack()
        
        self.root.mainloop()



