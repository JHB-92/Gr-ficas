import matplotlib.pyplot as plt

class grafica:

    def __init__(self, var_x = 'Variable x', var_y = 'Variable y', title = '', label_x = '', label_y = '', x_limit = (), y_limit =(), finish = 'False', save = False, label = '', color = ''):
        self.var_x = var_x
        self.var_y = var_y
        self.title = title
        self.label_x = label_x
        self.label_y = label_y
        self.save = save
        self.finish = finish
        self.x_limit = x_limit
        self.y_limit = y_limit
        self.label = label
        self.color = color

    def scatter(self):
        
        plt.scatter(self.var_x, self.var_y, label = self.label, color = self.color)
        self.__conditions__()
        self.__finish__()

    def line(self):
        plt.plot(self.var_x, self.var_y, label = self.label, color = self.color)
        self.__conditions__()
        self.__finish__()

    def fill_line(self):
        plt.plot(self.var_x, self.var_y, color = "black")
        plt.fill_between(self.var_x, self.var_y, label = self.label, color = self.color)
        self.__conditions__()
        self.__finish__()

    def __conditions__(self):
        plt.ylabel(self.label_y)
        plt.xlabel(self.label_x)
        plt.title(self.title)
        plt.legend()
        plt.grid(axis = 'y', color = 'gray', linestyle = 'dashed')
        plt.xlim(self.x_limit)
        plt.ylim(self.y_limit)

    def __save__(self):
        plt.savefig(self.title+".png", dpi = 800)
        plt.close()

    def __finish__(self):
        if self.save == True:
            self.__save__()
        if self.finish == True:
            plt.show()



