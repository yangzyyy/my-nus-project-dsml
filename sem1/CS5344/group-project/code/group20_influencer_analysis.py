import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

columns = ['tweet_id', 'username', 'timestamp', 'follower', 'friends', 'retweets', 'favorites', 'entities', 'sentiment', 'mentions', 'hashtag', 'urls']
data = pd.read_csv('TweetsCOV19.tsv', sep='\t', header=None, error_bad_lines=False, names=columns)
# sent_temp = data[['entities', 'sentiment']]
# sent_temp['text'] = sent_temp['entities'].map(lambda x: x.split(';'))\
#     .map(lambda x: ' '.join(x[i].split(':')[0] for i in range(len(x))))

tweet_per_user = []
for i in range(5, 51, 5):
    tweet_per_user.append(len(data[data['follower'] > 10000*i])/len(data[data['follower'] > 10000*i]['username'].unique()))
followers = [n for n in range(5, 51, 5)]
plt.plot(followers, tweet_per_user,'ro-', color='#4169E1', alpha=0.8)
plt.xlabel('number of followers')
plt.ylabel('average tweets per user')
plt.show()


fig, ax = plt.subplots(1, 3, figsize=(16, 8))
influencer_data = data[data['follower'] > 500000]
print(len(influencer_data))
influencer_data['sentiment_combine'] = influencer_data['sentiment'].map(lambda x: x.split(' ')).map(lambda x: int(x[0])+int(x[1]))
count_influ = influencer_data['sentiment_combine'].value_counts()
print(count_influ)
sns.barplot(x=count_influ.index, y=count_influ,ax = ax[0])
ax[0].set_xlabel('sentiment_combine')
ax[0].set_ylabel('Count')
ax[0].set_title('Sentiment_combine count for influencer')


original_data = data[(data['follower'] < 10000) & (data['follower'] > 100)]
print(len(original_data))
original_data['sentiment_combine'] = original_data['sentiment'].map(lambda x: x.split(' ')).map(lambda x: int(x[0])+int(x[1]))
count_orig = original_data['sentiment_combine'].value_counts()
sns.barplot(x=count_orig.index, y=count_orig, ax = ax[1])
ax[1].set_xlabel('sentiment_combine')
ax[1].set_ylabel('Count')
ax[1].set_title('Sentiment_combine count for intermediary people')


ordinary_data = data[data['follower'] < 100]
print(len(ordinary_data))
ordinary_data['sentiment_combine'] = ordinary_data['sentiment'].map(lambda x: x.split(' ')).map(lambda x: int(x[0])+int(x[1]))
count_ord = ordinary_data['sentiment_combine'].value_counts()
sns.barplot(x=count_ord.index, y=count_ord,ax = ax[2])
ax[2].set_xlabel('sentiment_combine')
ax[2].set_ylabel('Count')
ax[2].set_title('Sentiment_combine count for ordinary people')

plt.show()

# average hashtag of different groups of people
ordinary_data['hashtag'].where(ordinary_data['hashtag']!='null;', 0, inplace=True)
ordinary_data['tag_count'] = ordinary_data['hashtag'].astype(str).map(lambda x: 0 if x == '0' else x.split(' '))\
    .map(lambda x: 0 if x == 0 else len(x))
avg_tag_ord = ordinary_data['tag_count'].sum()/len(ordinary_data)

original_data['hashtag'].where(original_data['hashtag']!='null;', 0, inplace=True)
original_data['tag_count'] = original_data['hashtag'].astype(str).map(lambda x: 0 if x == '0' else x.split(' '))\
    .map(lambda x: 0 if x == 0 else len(x))
avg_tag_med = original_data['tag_count'].sum()/len(original_data)
stats.kruskal(np.array(ordinary_data['tag_count'].astype(float)), np.array(original_data['tag_count'].astype(float)))

influencer_data['hashtag'].where(influencer_data['hashtag']!='null;', 0, inplace=True)
influencer_data['tag_count'] = influencer_data['hashtag'].astype(str).map(lambda x: 0 if x == '0' else x.split(' '))\
    .map(lambda x: 0 if x == 0 else len(x))
avg_tag_inf = influencer_data['tag_count'].sum()/len(influencer_data)
stats.kruskal(np.array(original_data['tag_count'].astype(float)), np.array(influencer_data['tag_count'].astype(float)))

# average mentions of different groups of people
ordinary_data['mentions'].where(ordinary_data['mentions']!='null;', 0, inplace=True)
ordinary_data['mention_count'] = ordinary_data['mentions'].astype(str).map(lambda x: 0 if x == '0' else x.split(' '))\
    .map(lambda x: 0 if x == 0 else len(x))
avg_mention_ord = ordinary_data['mention_count'].sum()/len(ordinary_data)

original_data['mentions'].where(original_data['mentions']!='null;', 0, inplace=True)
original_data['mention_count'] = original_data['mentions'].astype(str).map(lambda x: 0 if x == '0' else x.split(' '))\
    .map(lambda x: 0 if x == 0 else len(x))
avg_mention_med = original_data['mention_count'].sum()/len(original_data)
stats.kruskal(np.array(ordinary_data['mention_count'].astype(float)), np.array(original_data['mention_count'].astype(float)))

influencer_data['mentions'].where(influencer_data['mentions']!='null;', 0, inplace=True)
influencer_data['mention_count'] = influencer_data['mentions'].astype(str).map(lambda x: 0 if x == '0' else x.split(' '))\
    .map(lambda x: 0 if x == 0 else len(x))
avg_mention_inf = influencer_data['mention_count'].sum()/len(influencer_data)
stats.kruskal(np.array(original_data['mention_count'].astype(float)), np.array(influencer_data['mention_count'].astype(float)))

# entity null ratio
inf_ent_ratio = len(influencer_data[influencer_data['entities'] == 'null;'])/len(influencer_data)
med_ent_ratio = len(original_data[original_data['entities'] == 'null;'])/len(original_data)
ord_ent_ratio = len(ordinary_data[ordinary_data['entities'] == 'null;'])/len(ordinary_data)


