import asyncio
import httpx
import json

async def test_api_search():
    fund_name = "Axis India Manufacturing Fund"
    print(f"Searching for: {fund_name}")
    
    async with httpx.AsyncClient() as client:
        all_funds = []
        page = 1
        while page <= 3:  # Check first 3 pages
            response = await client.get("http://34.122.133.139:4000/api/funds/", params={"page": page, "limit": 100})
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    funds = data.get("data", [])
                    all_funds.extend(funds)
                    print(f"Found {len(funds)} funds on page {page}")
                    if len(funds) < 100:  # Last page
                        break
                page += 1
            else:
                print(f"API request failed with status {response.status_code}")
                break
        
        print(f"Total funds found: {len(all_funds)}")
        
        # Find matching fund by name (case-insensitive search)
        matching_funds = []
        fund_name_lower = fund_name.lower()
        for fund in all_funds:
            scheme_name = fund.get("scheme_name", "").lower()
            if fund_name_lower in scheme_name or scheme_name in fund_name_lower:
                matching_funds.append(fund)
                print(f"Found matching fund: {fund.get('scheme_name')}")
        
        if matching_funds:
            fund_data = matching_funds[0]
            print(f"Selected fund: {fund_data.get('scheme_name')}")
            print(f"Fund data: NAV={fund_data.get('nav')}, Expense Ratio={fund_data.get('expense_ratio')}, Manager={fund_data.get('fund_manager')}")
        else:
            print(f"No matching funds found for '{fund_name}'")
            # Show some available funds for debugging
            print("Sample available funds:")
            for i, fund in enumerate(all_funds[:10]):
                print(f"  {i+1}. {fund.get('scheme_name')}")

if __name__ == "__main__":
    asyncio.run(test_api_search()) 