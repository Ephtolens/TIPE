import numpy as np
from random import random, randint

#Fonction utile ici pour les calculs soit Sigmoid et sa dérivée ou tangente hyperbolique et sa dérivée
#On vectorise toutes ces fonctions pour que cela fonctionne avec des tableaux numpy
sigmoid = np.vectorize(lambda x: (1/(1 + np.exp(-x))))
dSigmoid = np.vectorize(lambda x : x/(1.1-x) )
tanh = np.tanh
dTanh = np.vectorize(lambda x : 1/np.cosh(x)**2)
relu = np.vectorize(lambda x : 0. if x < 0 else x)
drelu = np.vectorize(lambda x : 0. if x < 0 else 1.)


#Vround et Vint sont des fonctions pour arrondir les outputs
Vround = np.vectorize(round)
Vint = np.vectorize(int)



class NeuralNetwork:
	''' Cet objet est un réseau de neurone modulaire qui résoudra vos probleme petits et grand (enfin pas trop grand quand meme )'''

	#Fonction que l'on n'utilise pas directement dans l'apprentissage mais qui peut être utile
	def mutate(x):
	# Mute la valeur de x
		if(random() < 0.1):
			offset = randint(-100,100)/1000
			newx = offset + x
			return newx
		else : return x


	mutate = np.vectorize(mutate)

	def mutateNN(self):
		self.biasIH = self.mutate(self.biasIH)
		self.biasHO = self.mutate(self.biasHO)
		self.weightIH = self.mutate(self.weightIH)
		self.weightHO = self.mutate(self.weightHO)

	def transposeCoVect(self,x):
		''' Transpose des CoVecteurs numpy en vecteur (seulement les vecteurs car pour les matrices numpy une fonction existe deja) exemple : [1,2,3] -> [[1],[2],[3]] '''
		row = 1
		col = np.size(x,0)
		r = np.array([np.zeros(row) for _ in range(col)])
		for j in range (col):
			r[j,0] = x[j]
		return r


	def __init__(self, input, hidden, output, learningRate, func):

		''' Initialise le réseau de neuronnes aves les données renseignées par l'utilisateur
		input: nb de neurones en entrée
		hidden: nb de neurones cachés
		output: nb de sortiesortie
		learningRate: vitesse d'aprentissage (0.1 c'est pas mal)
		func: spécifie la fonction d'activation à utiliser '''

		self.inputN  = input
		self.hiddenN = hidden
		self.outputN = output
		self.function = func
		self.weightIH = np.array([[random() for __ in range(hidden)] for _ in range(input)]) #Weight from Input to Hidden
		self.biasIH = np.array([random() for __ in range(hidden)])
		self.weightHO = np.array([[random() for __ in range(output)] for _ in range(hidden)]) #Weight from Hidden to Output
		self.biasHO = np.array([random() for __ in range(output)])
		self.lr = learningRate

		if(func == 'sigmoid'):
			self.activation = sigmoid
			self.derivate = dSigmoid
		elif(func == 'tanh'):
			self.activation = tanh
			self.derivate = dTanh
		elif(func == 'relu'):
			self.activation = relu
			self.derivate = drelu


	def train(self, inputs, targets):

		''' Entraine le réseau de neuronne avec des inputs et des outputs '''

		for i in range(inputs.shape[0]):

			#Détermine les valeurs des neurones cachés:
			hidden_inputs = np.dot(inputs[i],self.weightIH) + self.biasIH
			hidden_outputs = self.activation (hidden_inputs)


			#Détermine les valeurs des neurones de sorties:
			output_inputs = np.dot(hidden_outputs,self.weightHO) + self.biasHO
			outputs = self.activation(output_inputs)


			#Détermine l'erreur sur les neurones de sorties:
			output_error = targets[i]-outputs
			weightHOT = np.transpose(self.weightHO)

			#Détermine l'erreur sur les neurones cachés:
			hidden_error = np.dot(output_error,weightHOT)


			#Corrige les poids H->O:
			gradient_output = self.derivate(outputs)
			gradient_output *= output_error
			gradient_output *= self.lr

			self.biasHO += gradient_output

			hidden_outputs_T = self.transposeCoVect(hidden_outputs)
			deltaW_outputs = np.dot(hidden_outputs_T,[gradient_output])
			self.weightHO += deltaW_outputs


			#Corrige les poids I->H :
			gradient_hidden = self.derivate(hidden_outputs)
			gradient_hidden *= hidden_error
			gradient_hidden *= self.lr

			self.biasIH += gradient_hidden

			inputs_T = self.transposeCoVect(inputs[i])
			deltaW_hidden = np.dot(inputs_T,[gradient_hidden])
			self.weightIH += deltaW_hidden

	def query(self,inputs):
		''' Demande les outputs à partir d'inputs '''
		hidden_inputs = np.dot(inputs, self.weightIH) + self.biasIH
		hidden_outputs = self.activation(hidden_inputs)

		output_inputs = np.dot(hidden_outputs,self.weightHO) + self.biasHO
		outputs = self.activation(output_inputs)

		return(outputs)

	def perf(self,inputs,outputs):
		''' Calcule les performances du réseau de neuronne avec un set de données test différentes de celles qui on servis pour l'entrainement'''
		cpt = 0
		for i in range(inputs.shape[0]):
			if ((Vint(Vround(self.query(inputs[i]))) == outputs[i]).all()):
				cpt+=1
		return (cpt*100)/inputs.shape[0]

	def saveNN(self,name):
		print(strMat(self.weightIH))
		f = open(name + ".nn", 'w')
		f.write(str(self.inputN) + "\n")
		f.write(str(self.hiddenN) + "\n")
		f.write(str(self.outputN) + "\n")
		f.write(strMat(self.weightIH) + "\n")
		f.write(strMat(self.weightHO) + "\n")
		f.write(strVec(self.biasIH) + "\n")
		f.write(strVec(self.biasHO) + "\n")
		f.write(str(self.lr) + "\n")
		f.write(str(self.function) + "\n")


nn = NeuralNetwork(2,2,1,0.01,"tanh")
for i in range(20000):
	nn.train(np.array([0,1]), np.array([0]))
	nn.train(np.array([0,0]), np.array([1]))
	nn.train(np.array([1,0]), np.array([0]))
	nn.train(np.array([1,1]), np.array([1]))

print(nn.query(np.array([0,0])))
