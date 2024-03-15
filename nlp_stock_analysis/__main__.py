from GoogleNews import GoogleNews
from textblob import TextBlob
from openai import OpenAI
from transformers import pipeline

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required, but unused
)


def main():
    googlenews = GoogleNews(encode="utf-8")

    googlenews.get_news("NVDA stock")

    apple_news = googlenews.results()

    for article in apple_news:
        title = article["title"]
        date = article["date"]
        link = article["link"], end = "\n\n"
        print(article)
        # print(title)
        # print(get_sentiment(title))
        # print("\n")


def get_sentiment(text):
    """
    args:
        text (str): input string
    returns:
        sentiment (float): score of sentiment from -1 to 1
    """

    # blob = TextBlob(text)
    # return blob.sentiment
    # response = client.chat.completions.create(
    #     model="llama2",
    #     messages=[
    #         {"role": "system", "content": "You are a sentiment analysis assistant."},
    #         {
    #             "role": "user",
    #             "content": "Return a number between -1 and 1, 1 being positive, -1 being negative. Only return the number and nothing else. Return JSON object with value being a float",
    #         },
    #     ],
    #     response_format={"type": "json_object"},
    # )
    # return response.choices[0].message.content
    model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_id)
    return sentiment_pipeline(text)


if __name__ == "__main__":
    main()
