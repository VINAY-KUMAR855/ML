import numpy as np

x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

def gradient(X, y, epoch):
	# initialize weights
	w = b = 0
	n = len(y)
	lr = 0.8
	for i in range(epoch):
		# forward pass
		y_pred = x * w + b
		# calculate error
		cost = (1/n)*np.sum((y-y_pred)**2)
		# calculate gradients
		dw = -(2/n)*np.sum((y-y_pred)*x)
		db = -(2/n)*np.sum((y-y_pred))
		# update weights
		w = w-lr*dw
		b = b-lr*db
		if i%100==0:
			print(f"epoch:{i}, cost: {cost}")

gradient(x,y,500)
	
