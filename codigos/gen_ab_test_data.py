# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

fields = ['bucket', 'age', 'start_click', 'session_time', 'converted',
		  'ticket_price','shipping']


def clamp(n, smallest, largest):
	return round(max(smallest, min(n, largest)))


def gen_age_data(mean_age_1=35, std_age_1=5, mean_age_2=35, std_age_2=5, N1=None, N2=None):
	age_1 = list(map(clamp, np.random.normal(loc=mean_age_1,scale=std_age_1,size=N1),
		itertools.repeat(20, N1), itertools.repeat(80, N1)))
	age_2 = list(map(clamp, np.random.normal(loc=mean_age_2,scale=std_age_2,size=N2),
		itertools.repeat(20, N2), itertools.repeat(80, N2)))
	age = np.hstack([age_1, age_2])
	return age


def gen_start_click_data(prob_clicked_1=0.5, prob_clicked_2=0.5, N1=None, N2=None):
	clicked_1 = [np.random.binomial(1,p=prob_clicked_1) for i in range(N1)]
	clicked_2 = [np.random.binomial(1,p=prob_clicked_2) for i in range(N2)]
	clicked = np.hstack([clicked_1, clicked_2])
	return clicked


def gen_session_time_data(lam_1=10, lam_2=8, clicked=None):
	session_time = []
	for click in clicked:
		if click:
			lam = lam_1
		else:
			lam = lam_2
		session_time.append(np.random.poisson(lam=lam))
	return session_time


def gen_converted_data(prob_converted_1=0.2, prob_converted_2=0.2, clicked=None, bucket=None):
	converted = []
	for i, click in enumerate(clicked):
		if click:
			if bucket[i]:
				converted.append(np.random.binomial(1,p=prob_converted_2))
			else:
				converted.append(np.random.binomial(1,p=prob_converted_1))
		else:
			converted.append(0)
	return converted


def gen_ticket_data(mean_ticket=100, converted=None):
	ticket = [np.random.normal(loc=mean_ticket,scale=20) if i else 0 for i in converted]
	return ticket


def gen_shipping_data(mean_shipping=20, converted=None):
	shipping = [clamp(np.random.normal(loc=mean_shipping,scale=5), 5, 40) if i else 0 for i in converted]
	return shipping


class ABTest(object):
	def __init__(self, n_data=10000, prob_clicked_1=0.4, prob_clicked_2=0.4,
		mean_age_1=35, std_age_1=5, mean_age_2=35, std_age_2=5, lam_1=10, lam_2=8,
		mean_ticket=100, mean_shipping=15, prob_converted_1=0.2, prob_converted_2=0.2):
		self.n_data = n_data
		self.N1 = int(np.random.normal(loc=n_data,scale=100))
		self.N2 = int(np.random.normal(loc=n_data,scale=100))
		self.prob_clicked_1 = prob_clicked_1
		self.prob_clicked_2 = prob_clicked_2
		self.mean_age_1 = mean_age_1
		self.std_age_1 = std_age_1
		self.mean_age_2 = mean_age_2
		self.std_age_2 = std_age_2
		self.lam_1 = lam_1
		self.lam_2 = lam_2
		self.mean_ticket = mean_ticket
		self.mean_shipping = mean_shipping
		self.prob_converted_1 = prob_converted_1
		self.prob_converted_2 = prob_converted_2
		self.bucket = np.hstack([np.zeros(self.N1),np.ones(self.N2)])
		self.df = pd.DataFrame()
		self.df = self.gen_test_data()

	def gen_test_data(self):
		age = gen_age_data(N1=self.N1, N2=self.N2)
		clicked = gen_start_click_data(prob_clicked_1=self.prob_clicked_1,
			prob_clicked_2=self.prob_clicked_2, N1=self.N1, N2=self.N2)
		session_time = gen_session_time_data(clicked=clicked)
		bucket = np.hstack([np.zeros(self.N1),np.ones(self.N2)])
		converted = gen_converted_data(prob_converted_1=self.prob_converted_1,
			prob_converted_2=self.prob_converted_2, clicked=clicked,
			bucket=bucket)
		ticket = gen_ticket_data(converted=converted)
		shipping = gen_shipping_data(converted=converted)

		self.df['bucket'] = bucket
		self.df['age'] = age
		self.df['start_click'] = clicked
		self.df['session_time'] = session_time
		self.df['converted'] = converted
		self.df['ticket_price'] = ticket
		self.df['shipping'] = shipping

		self.df = self.df.sample(frac=1).reset_index(drop=True)
		return self.df

	def plot_hist(self, x):
		plt.hist(x)
		plt.show()
		return None

	def mean_bucket(self, bucket, field):
		assert bucket in [0,1], 'Valor do bucket deve ser [0,1]'
		assert field in fields, 'Valor do campo fora do intervalo {}'.format(str(fields))
		return self.df[self.df['bucket']==bucket][field].mean()

	def std_bucket(self, bucket, field):
		assert bucket in [0,1], 'Valor do bucket deve ser [0,1]'
		assert field in fields, 'Valor do campo fora do intervalo {}'.format(str(fields))
		return self.df[self.df['bucket']==bucket][field].std()

	def ctr_bucket(self, bucket):
		assert bucket in [0,1], 'Valor do bucket deve ser [0,1]'
		return self.df[self.df['bucket']==bucket]['start_click'].mean()

	def conv_rate_abs_bucket(self, bucket):
		assert bucket in [0,1], 'Valor do bucket deve ser [0,1]'
		return self.df[self.df['bucket']==bucket]['converted'].mean()

	def conv_rate_rel_bucket(self, bucket):
		assert bucket in [0,1], 'Valor do bucket deve ser [0,1]'
		return self.df[(self.df['start_click']==1) & (self.df['bucket']==bucket)]['converted'].mean()


class AATest(ABTest):
	def __init__(self, n_data=10000, prob_clicked_1=0.4, prob_clicked_2=0.4,
		mean_age_1=35, std_age_1=5, mean_age_2=35, std_age_2=5, lam_1=10, lam_2=8,
		mean_ticket=100, mean_shipping=20, prob_converted_1=0.2, prob_converted_2=0.22):

		ABTest.__init__(self, n_data=n_data, prob_clicked_1=prob_clicked_1, 
			prob_clicked_2=prob_clicked_2, mean_age_1=mean_age_1, std_age_1=std_age_1,
			mean_age_2=mean_age_2, std_age_2=std_age_2, lam_1=std_age_2, lam_2=lam_2,
			mean_ticket=mean_ticket, mean_shipping=mean_shipping, 
			prob_converted_1=prob_converted_1, prob_converted_2=prob_converted_2)
		
		
class ColorABTest(ABTest):
	def __init__(self, n_data=10000, prob_clicked_1=0.4, prob_clicked_2=0.44,
		mean_age_1=35, std_age_1=5, mean_age_2=35, std_age_2=5, lam_1=10, lam_2=8,
		mean_ticket=100, mean_shipping=20, prob_converted_1=0.2, prob_converted_2=0.2):

		ABTest.__init__(self, n_data=n_data, prob_clicked_1=prob_clicked_1, 
			prob_clicked_2=prob_clicked_2, mean_age_1=mean_age_1, std_age_1=std_age_1,
			mean_age_2=mean_age_2, std_age_2=std_age_2, lam_1=std_age_2, lam_2=lam_2,
			mean_ticket=mean_ticket, mean_shipping=mean_shipping, 
			prob_converted_1=prob_converted_1, prob_converted_2=prob_converted_2)


class ShippingABTest(ABTest):
	def __init__(self, n_data=10000, prob_clicked_1=0.4, prob_clicked_2=0.4,
		mean_age_1=35, std_age_1=5, mean_age_2=35, std_age_2=5, lam_1=10, lam_2=8,
		mean_ticket=100, mean_shipping=20, prob_converted_1=0.2, prob_converted_2=0.23):

		ABTest.__init__(self, n_data=n_data/prob_clicked_1, prob_clicked_1=prob_clicked_1,
			prob_clicked_2=prob_clicked_2, mean_age_1=mean_age_1, std_age_1=std_age_1,
			mean_age_2=mean_age_2, std_age_2=std_age_2, lam_1=std_age_2, lam_2=lam_2,
			mean_ticket=mean_ticket, mean_shipping=mean_shipping/2,
			prob_converted_1=prob_converted_1, prob_converted_2=prob_converted_2)
