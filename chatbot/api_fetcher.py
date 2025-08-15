import aiohttp

async def fetch_fund_details_from_api(question: str) -> dict:
    url = "http://34.122.133.139:4000/api/query"
    payload = {"text": question}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            if resp.status == 200:
                return await resp.json()
            return {}