# Analysis of Customer Reviews
> A friend of mine was working with some customer reviews for competitors to the business he was starting. He asked if I would be able to use any of my coding/analysis skills to make the process more efficient. I hadn't ever really done much with text data before, but with the help of google I was able to draft up these results for him in a few hours. 

**Notes:**

Sections to analyze 
- overall, feedback, cons

The goal is to try and pull out similar phrases or common value propositions to find trends of they key reasons customers buy and also what they may dislike (the cons)


```python
# load libraries
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from matplotlib import pyplot as plt
import seaborn as sns

from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from PIL import Image
import collections
from collections import Counter

```

```python
# create a list of stop words (filler words)
# nltk.download('stopwords')
stopWords = set(stopwords.words('english'))
```

# Load in the data

```python
# read in data from the excel file
raw_data = pd.read_excel("review_data2.xlsx", sheet_name='Bonusly')

# check dimensions and format
print(raw_data.shape)
raw_data.head(3)
```

    (1104, 13)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Title</th>
      <th>Industry/Employee Count</th>
      <th>Overall Rating</th>
      <th>ReviewSource__HalfUnitMarginDiv-lnjke6-1 2</th>
      <th>Review Title</th>
      <th>Overall</th>
      <th>Feedback</th>
      <th>Cons</th>
      <th>StarRating__Rating-sc-9jwzgg-1 4</th>
      <th>RatingContainer__Root-zgij78-0 5</th>
      <th>StarRating__Rating-sc-9jwzgg-1 5</th>
      <th>RatingContainer__Root-zgij78-0 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Jared M.</td>
      <td>NOC Manager</td>
      <td>Information Technology and Services, 11-50 emp...</td>
      <td>2020-05-05</td>
      <td>Source: Capterra</td>
      <td>“Zero Admin Overhead Employee Recogntion Platf...</td>
      <td>Feed the tool money and watch your employees g...</td>
      <td>Ease of Use: Bonusly is the pinnacle of set it...</td>
      <td>Bonusly could probably do to work on some addi...</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1</td>
      <td>John G.</td>
      <td>CTO &amp; Founder</td>
      <td>Information Technology and Services, 11-50 emp...</td>
      <td>2020-05-05</td>
      <td>Source: Capterra</td>
      <td>“Peer Recognition...on Autopilot ”</td>
      <td>Great!</td>
      <td>Can't say enough good things about Bonus.ly. T...</td>
      <td>That we didn't implement it sooner at our orga...</td>
      <td>2020-05-05</td>
      <td>Value for Money</td>
      <td>2020-05-05</td>
      <td>Likelihood to Recommend</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Yasser F.</td>
      <td>VP of Engineering</td>
      <td>Information Technology and Services, 51-200 em...</td>
      <td>2020-04-05</td>
      <td>Source: Capterra</td>
      <td>“A must have in any company”</td>
      <td>Bonusly accomplishes what I need which is real...</td>
      <td>- Easy to set up and use\n - Slack integration...</td>
      <td>- Weak rewards dashboard\n - Manual fulfillmen...</td>
      <td>2020-04-05</td>
      <td>Value for Money</td>
      <td>2020-05-05</td>
      <td>Likelihood to Recommend</td>
    </tr>
  </tbody>
</table>
</div>



```python
# raw_data.info()
```

```python
# raw_data['Industry'].unique()
```

The Overall Rating and Likelihood to Recommend were read in as dates, so I'll have to fix that.

## Data Cleaning

```python
def format_text(column, company_name):
    # drop punctuation and make everything lowercase
    column = column.str.replace('[^\w\s]','').str.lower()
    
    # remove company name since it shows up in all cases
    column = column.str.replace(company_name.lower(), '')
#     column = column.str.replace('nan', '')
    
    return column.astype(str)


# function to clean up the text data
def clean_data(data, company_name, cols_to_keep = []):
    
    # to prevent overwriting the original data
    data = data.copy()
    
    # get actual overall rating value instead of a date
    data['Rating_'] = data['Overall Rating'].dt.month
    
    # covert all text to lowercase, and drop punctuation
    data['Overall_'] = format_text(data['Overall'], company_name=company_name)
    data['Overall_'] = np.where(data['Overall_'] == 'nan', '', data['Overall_'])

    data['Feedback_'] = format_text(data['Feedback'], company_name=company_name)
    data['Feedback_'] = np.where(data['Feedback_'] == 'nan', '', data['Feedback_'])
    
    
    data['Cons_'] = format_text(data['Cons'], company_name=company_name)
    data['Cons_'] = np.where(data['Cons_'] == 'nan', '', data['Cons_'])
    
    data['Combined_'] = data['Overall_'] + " " + data['Feedback_'] + " " + data['Cons_']
    
    # polarity of the review (score between -1 and 1)
    data['Polarity_Overall'] = data['Overall_'].map(lambda text: TextBlob(text).sentiment.polarity)
    data['Polarity_Feedback'] = data['Feedback_'].map(lambda text: TextBlob(text).sentiment.polarity)
    data['Polarity_Cons'] = data['Cons_'].map(lambda text: TextBlob(text).sentiment.polarity)
    
    # the length of the overall review (the raw, unformatted reviews)
    data['review_len'] = data['Overall'].astype(str).apply(len)
    data['word_count_Overall'] = data['Overall'].apply(lambda x: len(str(x).split()))   
    data['word_count_Feedback'] = data['Feedback'].apply(lambda x: len(str(x).split()))    
    data['word_count_Cons'] = data['Cons'].apply(lambda x: len(str(x).split()))    

    
    if len(cols_to_keep) != 0:
        # only return the columns of interest
        data = data[cols_to_keep]
        
    
    return data
```

```python
df = clean_data(raw_data, company_name="Bonusly")
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Title</th>
      <th>Industry/Employee Count</th>
      <th>Overall Rating</th>
      <th>ReviewSource__HalfUnitMarginDiv-lnjke6-1 2</th>
      <th>Review Title</th>
      <th>Overall</th>
      <th>Feedback</th>
      <th>Cons</th>
      <th>StarRating__Rating-sc-9jwzgg-1 4</th>
      <th>...</th>
      <th>Feedback_</th>
      <th>Cons_</th>
      <th>Combined_</th>
      <th>Polarity_Overall</th>
      <th>Polarity_Feedback</th>
      <th>Polarity_Cons</th>
      <th>review_len</th>
      <th>word_count_Overall</th>
      <th>word_count_Feedback</th>
      <th>word_count_Cons</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Jared M.</td>
      <td>NOC Manager</td>
      <td>Information Technology and Services, 11-50 emp...</td>
      <td>2020-05-05</td>
      <td>Source: Capterra</td>
      <td>“Zero Admin Overhead Employee Recogntion Platf...</td>
      <td>Feed the tool money and watch your employees g...</td>
      <td>Ease of Use: Bonusly is the pinnacle of set it...</td>
      <td>Bonusly could probably do to work on some addi...</td>
      <td>NaT</td>
      <td>...</td>
      <td>ease of use  is the pinnacle of set it and for...</td>
      <td>could probably do to work on some additional ...</td>
      <td>feed the tool money and watch your employees g...</td>
      <td>0.000000</td>
      <td>0.234091</td>
      <td>0.31875</td>
      <td>205</td>
      <td>35</td>
      <td>65</td>
      <td>29</td>
    </tr>
    <tr>
      <td>1</td>
      <td>John G.</td>
      <td>CTO &amp; Founder</td>
      <td>Information Technology and Services, 11-50 emp...</td>
      <td>2020-05-05</td>
      <td>Source: Capterra</td>
      <td>“Peer Recognition...on Autopilot ”</td>
      <td>Great!</td>
      <td>Can't say enough good things about Bonus.ly. T...</td>
      <td>That we didn't implement it sooner at our orga...</td>
      <td>2020-05-05</td>
      <td>...</td>
      <td>cant say enough good things about  their aweso...</td>
      <td>that we didnt implement it sooner at our organ...</td>
      <td>great cant say enough good things about  their...</td>
      <td>0.800000</td>
      <td>0.224934</td>
      <td>0.00000</td>
      <td>6</td>
      <td>1</td>
      <td>184</td>
      <td>9</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Yasser F.</td>
      <td>VP of Engineering</td>
      <td>Information Technology and Services, 51-200 em...</td>
      <td>2020-04-05</td>
      <td>Source: Capterra</td>
      <td>“A must have in any company”</td>
      <td>Bonusly accomplishes what I need which is real...</td>
      <td>- Easy to set up and use\n - Slack integration...</td>
      <td>- Weak rewards dashboard\n - Manual fulfillmen...</td>
      <td>2020-04-05</td>
      <td>...</td>
      <td>easy to set up and use\n  slack integration\n...</td>
      <td>weak rewards dashboard\n  manual fulfillment ...</td>
      <td>accomplishes what i need which is realtime pe...</td>
      <td>0.149583</td>
      <td>0.433333</td>
      <td>-0.37500</td>
      <td>536</td>
      <td>89</td>
      <td>32</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 25 columns</p>
</div>



### A vast majority of the reviews are positive

```python
# how many of each rating?
df['Rating_'].value_counts()
```




    5    905
    4    179
    3     19
    2      1
    Name: Rating_, dtype: int64



```python
# how many of each rating as a percent?
df['Rating_'].value_counts() * 100/ df.shape[0]
```




    5    81.974638
    4    16.213768
    3     1.721014
    2     0.090580
    Name: Rating_, dtype: float64



## Looking at Review with Highest and Lowest Polarities

Looking at all of the reviews by hand would be quite tedious. Instead we can rank the reviews based on their sentiment, and then glance at the most polarized reviews. This will help to hone in on the biggest likes/disklikes of customers. 

I also added a filter to only include reviews that have a minimum word count (specified in the code). That way we are seeing reviews with more detail instead of reviews with only hype (e.g. "I love this!!!")

(Note that the sentiment polarity of a review is scored between -1 (negative sentiment) and 1 (positive sentiment)



#### Overall

```python
# function to look at most polar reviews
def view_most_polar_reviews(data, field, positive=True, n=7, min_words = 20):
    """
    This function grabs the most polar reviews (positive by default)
    To see negateive reviews change positive=False
    """
    order = not positive
    temp_df = data.loc[
    (data['word_count_'+field] >= min_words),
    ['Polarity_'+field,field]
    ].sort_values(by='Polarity_'+field, ascending=order).head(n)
    for row in range(temp_df.shape[0]):
        print("Polarity:",temp_df.iloc[row,0])
        print(temp_df.iloc[row,1])
        print('\n')
    return
    
```

```python
# most positive
view_most_polar_reviews(df, 'Overall', positive = True, n=7, min_words=20)
```

    Polarity: 1.0
    Ease of use and intergration with Slack. Our team loves to Bonusly out for a bday, anniversary, or someone excecuting excellently at our clients.
    
    
    Polarity: 0.9
    Bonusly is incredible for team building and morale. I think every company should offer bonusly. If you're a company that believes in a tool like bonusly, you are immediately saying "I care about my employees well being."
    
    
    Polarity: 0.85
    Good way to recognize others. Still figuring out how to use it best after ~2 months and manage points efficiently.
    
    
    Polarity: 0.8
    This application is a great way to recognize your peers within your company. Definitely a great addition to our company :)
    
    
    Polarity: 0.8
    Bonusly lets me go beyond thank you. It allows me to reward those who help our office run well and those who keep our customers happy.
    
    
    Polarity: 0.8
    I like using the rewards to purchase what I want and you have a great selection of gifts and donations.
    
    
    Polarity: 0.8
    I would recommend using bonusly as it is a great application for giving rewards to the individuals and boosting their confidence.
    
    


```python
# most negative
view_most_polar_reviews(df, 'Overall', positive = False, n=7, min_words=20)
```

    Polarity: -0.2916666666666667
    I have been an ardent fan of this software and have implemented in my group to track productivity and reward hard work.
    
    
    Polarity: -0.25
    This tool helps motivate people to interact and brings the shy people out of their shell, which helps with overall office collaboration.
    
    
    Polarity: -0.2333333333333333
    I have already recommended this software to a few organisations. All the benefits are already mentioned in the pros I listed earlier, so this is a useless question
    
    
    Polarity: -0.2
    I like all the features that I can review all my claimed rewards and have the option also to refund a reward that's not available or made by mistake
    
    
    Polarity: -0.19999999999999998
    -Ease of use
     
     -Web or mobile or slack options for giving bonusly
     
     -Number of rewards is crazy and very flexible
    
    
    Polarity: -0.19791666666666666
    The benefit is that the price point for the amount of users we have has been less expensive than other options of the same style.
    
    
    Polarity: -0.1875
    ability to reward people with little effort. This software also allows people to pick the reward of their choice, vs me just buying them a visa gift card.
    
    


#### Feedback

```python
# most positive
view_most_polar_reviews(df, 'Feedback', positive = True, n=7, min_words=20)
```

    Polarity: 1.0
    After almost two months of use, I have not found anything that I dislike about the software. The user experience is excellent.
    
    
    Polarity: 1.0
    This drives team bonding I think and comradery. We use it all the time at InVision when someone does something awesome
    
    
    Polarity: 0.9099999999999999
    - very good software to recognize and credit your colleagues and power users at work
     - helps in showing the appreciation for the help received
     - Ease to use
     - plenty of gift options to redeem
    
    
    Polarity: 0.8
    I do not see a con to this function, it is great to be recognized for the work you do day in and day out
    
    
    Polarity: 0.8
    I dont have any cons. I dont think there is anything that we dont like about the system. Everything has been great.
    
    
    Polarity: 0.8
    Whether you use the app, online, or browser extension.....it could not any easier! It's a great way to show appreciation.
    
    
    Polarity: 0.8
    I like that all employees get a say in who can get a bonus. It is a great way to help show your appreciation. Someone ran a report for you? Send them points. Someone covered while you were on PTO? Send some points. Its great.
    
    


```python
# most negative
view_most_polar_reviews(df, 'Feedback', positive = False, n=7, min_words=20)
```

    Polarity: -0.6
    The restriction of gifs to include used to be abled to be customized, but have since been restricted to a pre-set pool. Disappointing.
    
    
    Polarity: -0.5
    - Sometimes it becomes difficult to find the page where I can see the bonuses given / received for the month
    
    
    Polarity: -0.4
    The admin dashboard is not great. Searching for users to edit, or delete them is a pain when you have 600+ employees (and growing).
    
    
    Polarity: -0.4
    - Retrospectively adding an emoji in the text is a bit annoying 
     - When someone gives a working group a reward and people add on to that the initial giver is not included although they could have been part of the project as well
    
    
    Polarity: -0.4
    Expiration of Bonusly points. If I am out of the office for an extended period of time and not working as closely with my co-workers I hate that I can not retain my allotted monthly points for future use.
    
    
    Polarity: -0.4
    Not a fan of the forced hashtag, it feels cheesy. Update the reward catalog and alert users to the updates
    
    
    Polarity: -0.39000000000000007
    When you try to scroll through the home page to view who has been thanked it is very slow and tends to get stuck.
    
    


#### Cons

```python
# most positive
view_most_polar_reviews(df, 'Cons', positive = True, n=7, min_words=20)
```

    Polarity: 1.0
    This is petty...but GIFs take a lot of time to load and sometimes they don't at all. User error? How best to use?
    
    
    Polarity: 0.8
    I wish there was a way to cash out your bonus monetarily. I think also there should be a feature to win additional bonusly when you are awarded team leader in a category
    
    
    Polarity: 0.8
    None that I can think of but it would be great if the rewards could include websites like Nykaa and Myntra.
    
    
    Polarity: 0.8
    The formatting of text in the Bonus.ly website sometimes does not sync as great with the Slack app as we'd like.
    
    
    Polarity: 0.6
    Honestly, I don't know what to tell, hmm maybe some chrome extensions doesn't work here that will make the GIF button won't work
    
    
    Polarity: 0.6
    How I'm restricted from adding my own custom hashtags and that I'm enforced to use a hashtag every time. How my bonusly points don't get carried over, I forgot to give anyone any one month and a lot of people missed out.
    
    
    Polarity: 0.6
    I wish I had all the money in the world to give out to people! That being said, the software is great. I have heard there is a way to donate points to causes, however, have not figured it out yet. I would love to be able to donate earned points under the rewards redemption page.
    
    


```python
# most negative
view_most_polar_reviews(df, 'Cons', positive = False, n=7, min_words=20)
```

    Polarity: -0.5
    I would suggest to add Notes space to custom rewards so employees can mention for example what day they are requesting for time off. Sometimes it gets difficult to track.
    
    
    Polarity: -0.5
    I've purchased rewards and they have been unavailable but I was unable to find this until after I've purchased them.
    
    
    Polarity: -0.35
    Would like ability to post a prop that is anonymous. Sometimes its not good to let everyone know you are thanking someone
    
    
    Polarity: -0.30000000000000004
    That I am being forced to provide a comment in this section and I have no comments about the software
    
    
    Polarity: -0.30000000000000004
    Sometimes the features are slow to loud or difficult to find when updates occur. Also when there are duplicate names it can be difficult to figure out who is who for newcomers.
    
    
    Polarity: -0.3
    Integration with Slack could use some work - the usernames in Slack don't match Bonusly names which can be confusing.
    
    
    Polarity: -0.3
    I don't have any "cons" to share. We haven't had any negative experiences with the software, so I can't offer any feedback for this section.
    
    


### Grabbing the most common words

```python
def plot_most_common(field, n=15):
    all_words = " ".join(list(df[field])).lower().split()
    all_words = [word for word in all_words if word not in stopWords]

    word_counts_overall = Counter(all_words)
    word_counts_overall.most_common(n)

    word, count = zip(*word_counts_overall.most_common(n))
    plt.figure(figsize=(15,4))
    plt.title(f"Top {n} Most Common Words for {field}")
    plt.bar(word,count)
    plt.xticks(rotation = 90, fontsize = 13)
    plt.show()
    return
```

#### Overall

```python
plot_most_common('Overall_', n=30)
```


![png](/images/CustomerReviews_files/output_27_0.png)


#### Feedback

```python
plot_most_common('Feedback_', n=30)
```


![png](/images/CustomerReviews_files/output_29_0.png)


#### Cons

```python
plot_most_common('Cons_', n=30)
```


![png](/images/CustomerReviews_files/output_31_0.png)


## Using Word Charts to get a feel for the relative frequency of each word from the reviews


```python
# function to make a word diagram for each of the columns
def make_word_diagram(text_field):
    text = " ".join(row for row in text_field)
    
    print(f"Total Words: {len(text)}")
    
    # Generate a word cloud image
    wordcloud = WordCloud(stopwords=stopWords, background_color="white").generate(text)
    
    # Display the generated image:
    plt.figure(figsize=[30,15])
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    
    return  
```

### Word chart for all text from Overall, Feedback, and Cons

```python
# combined 
temp_df = df#[]

print(temp_df.shape)
make_word_diagram(temp_df['Combined_'])
```

    (1104, 25)
    Total Words: 351478



![png](/images/CustomerReviews_files/output_35_1.png)


### Word chart for all text from Overall

```python
# add a filter based on overall rating
temp_df = df#[]

print(temp_df.shape)
make_word_diagram(temp_df['Overall_'])
```

    (1104, 25)
    Total Words: 134379



![png](/images/CustomerReviews_files/output_37_1.png)


### Word chart for all text from Feedback

```python
temp_df = df[
    (df['Feedback_'] != 'nan')
]
print(temp_df.shape)
make_word_diagram(temp_df['Feedback_'])
```

    (1104, 25)
    Total Words: 145895



![png](/images/CustomerReviews_files/output_39_1.png)


### Word chart for all text from Cons

```python
temp_df = df#[]

print(temp_df.shape)
make_word_diagram(df['Cons_'])
```

    (1104, 25)
    Total Words: 71202



![png](/images/CustomerReviews_files/output_41_1.png)


```python
# focus in on non 5 star reviews
temp_df = df[
    (df['Rating_'] != 5)
]
make_word_diagram(temp_df['Cons_'])
```

    Total Words: 15327



![png](/images/CustomerReviews_files/output_42_1.png)


### Scatter Plots of Polarity, Ratings, and Review Length

```python
plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
sns.scatterplot(x = 'Rating_', y = 'Polarity_Overall', data = df)
plt.title("Overall Rating vs Polarity")
plt.hlines(y=0, xmin=3, xmax=5, linestyles='--')

plt.subplot(1,2,2)
sns.scatterplot(x = 'review_len', y = 'Polarity_Overall', data = df)
plt.title("Review Length vs Polarity")
plt.hlines(y=0, xmin=0, xmax=2000, linestyles='--')
plt.show()
```


![png](/images/CustomerReviews_files/output_44_0.png)


```python
# distribution of overall polarity
plt.hist(df['Polarity_Overall'])
plt.show()
```


![png](/images/CustomerReviews_files/output_45_0.png)


### End of Notebook
