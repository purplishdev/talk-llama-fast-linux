import os
import pprint

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
#from fastapi.responses import FileResponse,StreamingResponse
import uvicorn

#from langchain.tools import Tool
#from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.utilities import GoogleSerperAPIWrapper

os.environ["GOOGLE_CSE_ID"] = "your_key"                           # 100 requests per day
os.environ["GOOGLE_API_KEY"] = "your_key"    #
os.environ["SERPER_API_KEY"] = "your_key"   # 2500 requests total. https://serper.dev/api-key





app = FastAPI()
# Add CORS middleware 
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/google")
def google(q: str):
    print("Googling: "+q)
    if q:
        try:
            
            #search = GoogleSerperAPIWrapper( gl="ru", hl="ru")
            search = GoogleSerperAPIWrapper( gl="us", hl="en") #gl = geo country, hl = locale
            #print(search.run("Прогноз погоды в сочи"))
            result = search.run(q)
            return result # plain text
            
        except ValueError as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=500, detail=str("Error: empty search query"))

#search = GoogleSearchAPIWrapper()
#search = GoogleSearchAPIWrapper(k=1)

#tool = Tool(
#    name="google_search",
#    description="Search Google for recent results.",
#    func=search.run,
#)

#def top5_results(query):
#    return search.results(query, 5)


#tool2 = Tool(
#    name="Google Search Snippets",
#    description="Search Google for recent results.",
#    func=top5_results,
#)

#print(tool.run("How old is Keanu Reeves?"))



        
if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port=8003)