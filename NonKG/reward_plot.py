import numpy as np
import matplotlib.pyplot as plt


item_reward = np.load('./ml-1m_reward_item.npy')
user_reward = np.load('./ml-1m_reward_user.npy')



plt.plot(np.arange(len(item_reward)), item_reward)


plt.plot(np.arange(len(user_reward)), user_reward)
