from bs4 import BeautifulSoup
import requests, json

url = "https://photographyweek.tumblr.com/archive"
# response = requests.get(url)
with open("../data/tumblr_photographyweek.html") as infile:
    text = infile.read()
soup = BeautifulSoup(text, "html.parser")

urls = []
for link in soup.find_all("a"):
    url = link.get("href")
    urls.append(url)

results = []
for url in urls[4:]:
    response = requests.get(url+"/amp")
    s = BeautifulSoup(response.text, "html.parser")
    text = s.find_all("p")
    if text and len(text)>2:
        title = text[0].text
        if title == "":
            title = text[1].text
            content = text[2].text
        else:
            content = text[1].text
        if text != "":
            result = {"title": title, "text": title + "\n" + content, "url": url}
            results.append(result)

outfile = "../data/tumblr_photographyweek.jsonl"
with open(outfile, "w") as outf:
    for result in results:
        json.dump(result, outf)
        outf.write("\n")
