# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from Queue import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import os.path

import random,util,math,numpy
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Convolution2D, Flatten
from keras.optimizers import SGD


class QLearningAgent(ReinforcementAgent):
   
	def __init__(self, **args):
		"You can initialize Q-values here..."
		ReinforcementAgent.__init__(self, **args)

	def getFeatures(self, state, action):
		
		# ghost proximity
		ghostDistance = float(self.getGhostDistance(state, action))
		ghost = self.getClosestGhost(state, action)
		
		# pill proximity
		foodDistance = float(self.getFoodDistance(state, action))

		# disjunction proximity
		intersectionDistance = float(self.getIntersectionDistance(state, action))

		# capsule proximity
		capsuleDistance = float(self.getCapsuleDistance(state, action))
	
		# total number of food collected
		progress = 1.0 - float(state.generateSuccessor(0, action).getNumFood()) / float(self.totalFoodCount)

		# create input vector as a column numpy array
		x = numpy.zeros((1,self.inputDim))

		x[0][0] = 1 / ghostDistance
		x[0][1] = 1 / foodDistance
		x[0][2] = 1 / intersectionDistance
		x[0][3]	= 1 / capsuleDistance
		x[0][4] = progress
		
		# 1 if pacman is getting closer to its gloabally closest food when moving in a direction, 0 otherwise
		if  action != Directions.STOP and self.getFoodDistance(state, action) - 1 < self.getFoodDistance(state, Directions.STOP):
			x[0][5] = 1
		
		# 1 if the closest ghost in this direction is scared, 0 otherwise
		if ghost != None:
			if ghost.scaredTimer > 0:
				x[0][6] = 1

		# if no ghost in this direction was found, do the same for the globally closest ghost instead
		# we assume that we will guarantee to find a ghost when action == directions.STOP
		else:
			
			ghost = self.getClosestGhost(state, Directions.STOP)
			if ghost != None and ghost.scaredTimer > 0:
				x[0][6] = 1
	
		return x

	def getQValue(self, state, action):
		return self.network.predict(self.getFeatures(state, action))[0][0]		

	# finds the legal action in @state that corresponds to the highest Q-value and returns it
	def computeValueFromQValues(self, state):
		return max([self.getQValue(state, a) for a in self.getLegalActions(state)])

	# finds and returns the legal action in @state that corresponds to the highest Q-value
	def computeActionFromQValues(self, state):

		values = [self.getQValue(state, a) for a in self.getLegalActions(state)]
		maxValue = max(values)
		return random.choice([a for a,v in zip(actions, values) if v == maxValue])

	def debugInput(self, state, actions):

		print('======================================')
		for a in actions:

			# ghost proximity
			ghostDistance = self.getGhostDistance(state, a)
			ghost = self.getClosestGhost(state, a)
		
			# pill proximity
			foodDistance = self.getFoodDistance(state, a)

			# disjunction proximity
			intersectionDistance = self.getIntersectionDistance(state, a)

			# capsule proximity
			capsuleDistance = self.getCapsuleDistance(state, a)
	
			# total number of food collected
			progress = 1.0 - float(state.generateSuccessor(0, a).getNumFood()) / float(self.totalFoodCount)

			closerToGlobalFood = 0
			if  a != Directions.STOP and self.getFoodDistance(state, a) - 1 < self.getFoodDistance(state, Directions.STOP):
				closerToGlobalFood = 1
		
			scaredGhost = 0
			# 1 if the closest ghost in this direction is scared, 0 otherwise
			if ghost != None:
				if ghost.scaredTimer > 0:
					scaredGhost = 1

			# if no ghost in this direction was found, do the same for the globally closest ghost instead
			# we assume that we will guarantee to find a ghost when action == directions.STOP
			else:
			
				ghost = self.getClosestGhost(state, Directions.STOP)
				if ghost != None and ghost.scaredTimer > 0:
					scaredGhost = 1

			x = self.getFeatures(state, a)
			y = self.network.predict(x)[0][0]
			print 'a: %5s | x = [%3d, %3d, %3d, %3d, %.2f, %1d, %1d] | out = %.4f' % (a, ghostDistance, foodDistance, intersectionDistance, capsuleDistance, progress, closerToGlobalFood, scaredGhost, y)	

		p = ""
		i = 1
		for ghostPos in state.getGhostPositions():
			print "ghost %d is at position (%2d, %2d)" % (i, ghostPos[0], ghostPos[1])
			i += 1

				
			
		#raw_input("press to continue")

	def getAction(self, state):
		"""
		  Compute the action to take in the current state.	With
		  probability self.epsilon, we should take a random action and
		  take the best policy action otherwise.  Note that if there are
		  no legal actions, which is the case at the terminal state, you
		  should choose None as the action.
		"""

		# just doing what the text above says ...
		if state.isWin() or state.isLose():
			return None

		actions = self.getLegalActions(state)
	
		# calculate the exploration rate
		self.expRate = self.getExplorationRate(state)

		# perform a random action with prob. self.expRate
		if random.uniform(0, 1) <= self.expRate and self.explore:
			return random.choice(actions)

		# make move according to network output
		values = [self.getQValue(state, a) for a in actions]
		maxValue = max(values)
		
		# record the maximum Q-value of this state to generate average Q-values at terminal state
		self.qHistory.append(maxValue)

		#self.debugInput(state, actions)

		return random.choice([a for a, v in zip(actions, values) if v == maxValue])

	def update(self, state, action, nextState, reward):
		"""
		  The parent class calls this to observe a
		  state = action => nextState and reward transition.
		  You should do your Q-Value update here

		  NOTE: You should never call this function,
		  it will be called on your behalf
		"""
		
		# used to print the total number of turns at the terminal state
		self.turn += 1

		if self.trainOnline:


			# --- Experience recording ---

			# record the performed move
			pattern = self.getFeatures(state, action)
			target = reward

			# add additional term to the target in accordance with to Q-learning theory
			if not nextState.isWin() and not nextState.isLose():
				target = target + self.discount * self.computeValueFromQValues(state)

			# save observed transition in the experience replay list
			transition = (pattern, target)
			self.experiences.append(transition)


			# --- Experience replay training ---

			# only train of you have enough experiences
			if len(self.experiences) >= self.minimumExperienceSize:

				# create training set pair
				patterns = numpy.random.random((self.batchSize, self.inputDim))
				targets = numpy.random.random((self.batchSize, 1))
		
				# sample random transitions from previous experiences and use them as a sample points
				for i in range(self.batchSize):
				
					r = random.randint(0, len(self.experiences) - 1)

					patterns[i] = self.experiences[r][0]
					targets[i][0] = self.experiences[r][1]

				# train network
				if self.verboseTraining:
					self.network.fit(patterns, targets, nb_epoch = 1, batch_size = self.batchSize, verbose = 1)
				else:
					self.network.fit(patterns, targets, nb_epoch = 1, batch_size = self.batchSize, verbose = 0)

		
				# renew experiences when maximum capacity has been reached
				if len(self.experiences) > self.maximumExperienceSize:
					self.experiences.pop(0)

		if self.displayWeights:
				print self.network.get_weights()


	# calculates the correct exploration rate depending on method of choice
	def getExplorationRate(self, state):

		# if decay exploration on episode, linearly interpolate the accurate exploration rate
		if self.episodesSoFar <= self.decayPeriod and self.decayExplorationOnEpisode:

			fraction = float(self.episodesSoFar) / float(self.decayPeriod)
			return self.initialExplorationRate -(self.initialExplorationRate - self.finalExplorationRate) * fraction
		else:
			return self.finalExplorationRate


	def getPolicy(self, state):
		return self.computeActionFromQValues(state)

	def getValue(self, state):
		return self.computeValueFromQValues(state)

	def getFoodDistance(self, state, action):

		# get the relevant positions
		previousPosition = state.getPacmanState().getPosition()
		newPosition = state.generateSuccessor(0, action).getPacmanState().getPosition()

		# setup data structures for the search
		openQueue = Queue()
		closedSet = set()

		# ensures we 'close of our back' and start searching only in the given direction
		if action != Directions.STOP:

			closedSet.add(previousPosition)

			root = self.nodes[newPosition[0]][newPosition[1]]
			root.distance = 1
			openQueue.put(root)

		else:
			root = self.nodes[previousPosition[0]][previousPosition[1]]
			root.distance = 1
			openQueue.put(root)
			
		# start the BF search
		while openQueue.qsize() > 0:

			# pop new node and add it to the closed set
			currentNode = openQueue.get()
			closedSet.add(currentNode.position)

			# check for our objective
			if state.hasFood(currentNode.x, currentNode.y):
				return currentNode.distance

			for candidateNode in currentNode.neighbors:
				
				# increment distance for every edge you cross in the node-grid-graph
				candidateNode.distance = currentNode.distance + 1

				if not candidateNode.position in closedSet:
					openQueue.put(candidateNode)


		# no food was found in this direction
		return self.maxDistance

	def getGhostDistance(self, state, action):

		# get the relevant positions
		previousPosition = state.getPacmanState().getPosition()
		newPosition = state.generateSuccessor(0, action).getPacmanState().getPosition()

		# setup data structures for the search
		openQueue = Queue()
		closedSet = set()

		# ensures we 'close of our back' and start searching only in the given direction
		if action != Directions.STOP:

			closedSet.add(previousPosition)

			root = self.nodes[newPosition[0]][newPosition[1]]
			root.distance = 1
			openQueue.put(root)

		else:
			root = self.nodes[previousPosition[0]][previousPosition[1]]
			root.distance = 1
			openQueue.put(root)
			
		# start the BF search
		while openQueue.qsize() > 0:

			# pop new node and add it to the closed set
			currentNode = openQueue.get()
			closedSet.add(currentNode.position)

			# check for our objective
			for ghostPosition in state.getGhostPositions():

				# round ghost positions in case they are inbetween nodes
				# this happens when they are scared and have reduced movement speed
				if round(ghostPosition[0]) == currentNode.position[0]:
					if round(ghostPosition[1]) == currentNode.position[1]:
						return currentNode.distance
			

			for candidateNode in currentNode.neighbors:
				
				# increment distance for every edge you cross in the node-grid-graph
				candidateNode.distance = currentNode.distance + 1

				if not candidateNode.position in closedSet:
					openQueue.put(candidateNode)


		# No ghost was found in this direction
		return self.maxDistance

	def getCapsuleDistance(self, state, action):

		# get the relevant positions
		previousPosition = state.getPacmanState().getPosition()
		newPosition = state.generateSuccessor(0, action).getPacmanState().getPosition()

		# setup data structures for the search
		openQueue = Queue()
		closedSet = set()

		# ensures we 'close of our back' and start searching only in the given direction
		if action != Directions.STOP:

			closedSet.add(previousPosition)

			root = self.nodes[newPosition[0]][newPosition[1]]
			root.distance = 1
			openQueue.put(root)

		else:
			root = self.nodes[previousPosition[0]][previousPosition[1]]
			root.distance = 1
			openQueue.put(root)
			
		# start the BF search
		while openQueue.qsize() > 0:

			# pop new node and add it to the closed set
			currentNode = openQueue.get()
			closedSet.add(currentNode.position)

			# check for our objective
			for capsule in state.getCapsules():
				if capsule == currentNode.position:
					return currentNode.distance

			for candidateNode in currentNode.neighbors:
				
				# increment distance for every edge you cross in the node-grid-graph
				candidateNode.distance = currentNode.distance + 1

				if not candidateNode.position in closedSet:
					openQueue.put(candidateNode)


		# No capsule was found in this direction
		return self.maxDistance

	def getIntersectionDistance(self, state, action):

		# get the relevant positions
		previousPosition = state.getPacmanState().getPosition()
		newPosition = state.generateSuccessor(0, action).getPacmanState().getPosition()

		# setup data structures for the search
		openQueue = Queue()
		closedSet = set()

		# ensures we 'close of our back' and start searching only in the given direction
		if action != Directions.STOP:

			closedSet.add(previousPosition)

			root = self.nodes[newPosition[0]][newPosition[1]]
			root.distance = 1
			openQueue.put(root)

		else:
			root = self.nodes[previousPosition[0]][previousPosition[1]]
			root.distance = 1
			openQueue.put(root)
			
		# start the BF search
		while openQueue.qsize() > 0:

			# pop new node and add it to the closed set
			currentNode = openQueue.get()
			closedSet.add(currentNode.position)

			# check for our objective
			if len(currentNode.neighbors) >= 3:
				return currentNode.distance

			for candidateNode in currentNode.neighbors:
				
				# increment distance for every edge you cross in the node-grid-graph
				candidateNode.distance = currentNode.distance + 1

				if not candidateNode.position in closedSet:
					openQueue.put(candidateNode)


		# No intersection was found in this direction
		return self.maxDistance


	def getClosestGhost(self, state, action):

		# get the relevant positions
		previousPosition = state.getPacmanState().getPosition()
		newPosition = state.generateSuccessor(0, action).getPacmanState().getPosition()

		# setup data structures for the search
		openQueue = Queue()
		closedSet = set()

		# ensures we 'close of our back' and start searching only in the given direction
		if action != Directions.STOP:

			closedSet.add(previousPosition)

			root = self.nodes[newPosition[0]][newPosition[1]]
			root.distance = 1
			openQueue.put(root)

		else:
			root = self.nodes[previousPosition[0]][previousPosition[1]]
			root.distance = 1
			openQueue.put(root)
			
		# start the BF search
		while openQueue.qsize() > 0:

			# pop new node and add it to the closed set
			currentNode = openQueue.get()
			closedSet.add(currentNode.position)

			# check for our objective
			# round ghost positions in case they are inbetween nodes
			# this happens when they are scared and have reduced movement speed
			for ghost in state.getGhostStates():
				 if round(ghost.getPosition()[0]) == currentNode.position[0]:
					if round(ghost.getPosition()[1]) == currentNode.position[1]:
						return ghost

			for candidateNode in currentNode.neighbors:
				
				# increment distance for every edge you cross in the node-grid-graph
				candidateNode.distance = currentNode.distance + 1

				if not candidateNode.position in closedSet:
					openQueue.put(candidateNode)


		# No ghost was found in this direction
		return None



class PacmanQAgent(QLearningAgent):
	"Exactly the same as QLearningAgent, but with different default parameters"

	def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
		"""
		These default parameters can be changed from the pacman.py command line.
		For example, to change the exploration rate, try:
			python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

		alpha	 - learning rate
		epsilon  - exploration rate
		gamma	 - discount factor
		numTraining - number of training episodes, i.e. no learning after these many episodes
		"""
		args['gamma'] = gamma
		args['alpha'] = alpha
		args['numTraining'] = numTraining
		self.index = 0	# This is always Pacman

			# data collection
		# number of games that we won
		self.wins = 0
		
			# BFS settings
		# returns maxDistance if BFS is unsuccessful in finding objective
		self.maxDistance = 400

			# exploration settings
		# determines if exploration of any policy should be performed or not
		self.explore = False

		# exploration rate that the model will start with
		self.initialExplorationRate = 1.0

		# exploration rate that the model will end with after @decayPeriod episodes
		# if @decayExplorationOnEpisode is False, the exploration rate will be set to this one for all episodes
		self.finalExplorationRate = 0.01

		# controls if the exploration rate should decrease for each episode
		self.decayExplorationOnEpisode = False

		# number of games in which the exploration rate decreases linearly from @initialExplorationRate to @finalExplorationRate
		self.decayPeriod = 20

			# training settings
		# determines if pacman will be training between turns or not
		self.trainOnline = False	
	
		# controls if loss values for each training session should print to screen
		self.verboseTraining = False		

		# controls if the weights of the network should be printed out after each training session
		self.displayWeights = False	

		# size of training batch for each training session
		self.batchSize = 32				

			# experience replay settings

		# list that contain all transitions that pacman has made			
		self.experiences = []

		# maximum number of transitions the experience list can hold
		# exceeding it will completely clear @experiences
		self.maximumExperienceSize = 10000

		# minimum number of transitions needed to start a training session
		# if having less, no training will be done for this turn
		self.minimumExperienceSize = 32

			# network settings
		# network dimensions
		self.inputDim = 7	
		self.outputDim = 1
		self.hiddenUnits = 10	

		# constructs a network if there is none
		if not os.path.isfile('./network.json'):

			print 'Initializing network ...'
			self.network = Sequential()

			# specify input layer and add a hidden layer, add activation function and a single neural output layer
			self.network.add(Dense(self.hiddenUnits, input_dim=self.inputDim, init="uniform"))
			self.network.add(Activation('relu'))
			self.network.add(Dense(self.outputDim, init="uniform"))

			# define @optimizer and compile the network
			sgd = SGD(lr=0.0002, decay=1e-6, momentum=0.95, nesterov=True)
			self.network.compile(loss='mean_squared_error', optimizer=sgd)
			self.network.summary()

			# save network and weights
			json_string = self.network.to_json()
			open('network.json', 'w').write(json_string)
			self.network.save_weights('weights.h5')

		else:
			print 'Loading network ...'
			self.network = model_from_json(open('network.json').read())

			sgd = SGD(lr=0.0002, decay=1e-6, momentum=0.95, nesterov=True)
			self.network.load_weights('weights.h5')
			self.network.compile(loss='mean_squared_error', optimizer=sgd)
			self.network.summary()

		print 'Network is ready to use.'
		QLearningAgent.__init__(self, **args)

	def getAction(self, state):
		"""
		Simply calls the getAction method of QLearningAgent and then
		informs parent of action for Pacman.  Do not change or remove this
		method.
		"""
		action = QLearningAgent.getAction(self,state)
		self.doAction(state,action)
		return action



class ApproximateQAgent(PacmanQAgent):
	"""
	   ApproximateQLearningAgent

	   You should only have to overwrite getQValue
	   and update.	All other QLearningAgent functions
	   should work as is.
	"""
	def __init__(self, extractor='IdentityExtractor', **args):
		self.featExtractor = util.lookup(extractor, globals())()
		PacmanQAgent.__init__(self, **args)
		self.weights = util.Counter()

	def getWeights(self):
		return self.weights

	def getQValue(self, state, action):
		"""
		  Should return Q(state,action) = w * featureVector
		  where * is the dotProduct operator
		"""
		"*** YOUR CODE HERE ***"
		util.raiseNotDefined()

	def update(self, state, action, nextState, reward):
		"""
		   Should update your weights based on transition
		"""
		"*** YOUR CODE HERE ***"
		util.raiseNotDefined()

	def final(self, state):
		"Called at the end of each game."
		# call the super-class final method
		PacmanQAgent.final(self, state)

		# did we finish training?
		if self.episodesSoFar == self.numTraining:
			# you might want to print your weights here for debugging
			"*** YOUR CODE HERE ***"
			pass
