from tkinter import *
from PIL import ImageTk, Image
import os, time
import random

correctPolicy = [0,1,2,3,4,5]

agentPolicies = [[0,2,5],
                 [0,3,5],
                 [1,5,4]]

def createGUI(agentPolicies, optimalPolicy):

    imgFiles = ['a.png',
                'b.png',
                'c.png',
                'd.png',
                'pt.png',
                'vt.png']

    root = Tk()
    actionImgs = []

    for filename in imgFiles:
        image = Image.open(filename)
        image = image.resize((100, 100), Image.ANTIALIAS) #The (250, 250) is (height, width)
        img = ImageTk.PhotoImage(image)
        actionImgs.append(img)

    def getActionImages(policy, actionImages):
        imgs = []

        for element in policy:
            imgs.append(actionImages[element])

        return imgs

    correctImgs = getActionImages(correctPolicy, actionImgs)

    root.configure(bg = 'grey')

    text1 = Label(root, text = 'Optimal Actions: ', bg = 'grey', font = 18).grid(row = 1, column = 0)

    text2 = Label(root, text = 'Agent\'s Actions: ', bg = 'grey', font = 18).grid(row = 2, column = 0)

    iterationNumber = Label(root, text = 'Iteration Value: ', bg = 'grey', font = 18).grid(row = 3, column = 0)

    iterationValue = Label(root, text = '0', bg = 'grey', font = 18, height = 5)

    iterationValue.grid(row = 3, column = 1)

    i = 1
    for image in correctImgs:
        Label(root, image = image, borderwidth = 10, bg = 'grey').grid(row = 1,column = i)
        i += 1

    lab0 = Label(root, image = None, borderwidth = 10, bg = 'grey')
    lab0.grid(row = 2,column = 1)

    lab1 = Label(root, image = None, borderwidth = 10, bg = 'grey')
    lab1.grid(row = 2,column = 2)

    lab2 = Label(root, image = None, borderwidth = 10, bg = 'grey')
    lab2.grid(row = 2,column = 3)

    lab3 = Label(root, image = None, borderwidth = 10, bg = 'grey')
    lab3.grid(row = 2,column = 4)

    lab4 = Label(root, image = None, borderwidth = 10, bg = 'grey')
    lab4.grid(row = 2,column = 5)

    lab5 = Label(root, image = None, borderwidth = 10, bg = 'grey')
    lab5.grid(row = 2,column = 6)

    def getNewPolicy(agentPolicies = agentPolicies):
        if (int(iterationValue['text'])-1) < len(agentPolicies):

            return agentPolicies[int(iterationValue['text'])-1]
        else:
            return None

    def advance(agentPolicy = None, actionImgs = actionImgs):

        iterationValue['text'] = str(int(iterationValue['text'])+1)

        agentPolicy = getNewPolicy()

        if agentPolicy is None:
            time.sleep(5)
            exit()

        def getActionImages(policy, actionImages):
            imgs = ['','','','','','']

            # for element in policy:
            #     imgs.append(actionImages[element])

            # for element in range(0, (6-len(policy))):
            #     imgs.append('')

            for element in policy:
                imgs[element] = actionImages[element]

            return imgs

        agentImgs = getActionImages(agentPolicy, actionImgs)

        root.update_idletasks()

        lab0.config(image = agentImgs[0], bg = 'grey')
        lab0.image = agentImgs[0]

        lab1.config(image = agentImgs[1], bg = 'grey')
        lab1.image = agentImgs[1]

        lab2.config(image = agentImgs[2], bg = 'grey')
        lab2.image = agentImgs[2]

        lab3.config(image = agentImgs[3], bg = 'grey')
        lab3.image = agentImgs[3]

        lab4.config(image = agentImgs[4], bg = 'grey')
        lab4.image = agentImgs[4]

        lab5.config(image = agentImgs[5], bg = 'grey')
        lab5.image = agentImgs[5]

        root.update_idletasks()
        root.after(200, advance)

    root.after(200, advance)
    root.mainloop()


createGUI(agentPolicies = agentPolicies, optimalPolicy=correctPolicy)
