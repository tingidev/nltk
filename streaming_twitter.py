from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s

#consumer key, consumer secret, access token, access secret.
ckey="GbOWNQoLB7lVDw1i6HHIwM67U"
csecret="6pHy4x5pta9hFPpZqFz5liTP0RHWKAmKSgCYlBPBh9rG4eLC97"
atoken="799212607798448128-pXAWymbnsJySTdRjuSuZczC9MZmnCe6"
asecret="4ua2WY601TQO2k7nO0YDoqyY5wqf9TEg49b8K0XKhGEC3"

class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)

        tweet = ascii(all_data["text"])
        sent_val, sent_conf = s.sentiment(tweet)
        print(tweet, sent_val, sent_conf)

        if sent_conf*100 >= 80:
            output = open('C:/Users/jvw/Documents/Python Scripts/nltk/twitter-out.txt','a')
            output.write(sent_val)
            output.write('\n')
            output.close()

        return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["happy"])
