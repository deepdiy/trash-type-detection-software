
import requests
url='https://nax65i26tfbgvn3xbvjhm3c3ci.appsync-api.us-west-2.amazonaws.com/graphql'

# json={
#   "input": {
#     "predictedWasteClass": "bottle",
#     "actualWasteClass": null,
#     "createdAt": "2011-05-01T17:10:00z",
#     "imageUrl": "asadsd",
#     "geoTag": "test",
#     "xonecoor": 1,
#     "yonecoor": 2 ,
#     "xtwocoor": 3,
#     "ytwocoor": 4
#   }
# }
api_token='da2-kgl545mdabdjxezl7pm7gf3c3y'

payload = {
      "predictedWasteClass": "test_python_api",
      "actualWasteClass": '',
      "createdAt": "2011-05-01T17:10:00z",
      "imageUrl": "asadsd",
      "geoTag": "test",
      "xonecoor": 1,
      "yonecoor": 2 ,
      "xtwocoor": 3,
      "ytwocoor": 4
}

json = {'mutation':'''
CreateWaste($input: CreateWasteInput!) {
  createWaste(input: {}) {

    predictedWasteClass: "{}"，
    actualWasteClass: "{}",
    createdAt: "{}",
    imageUrl: "{}",
    geoTag: "{}",
    xonecoor: {},
    yonecoor: {},
    xtwocoor: {},
    ytwocoor: {},
  }
}
'''.format(payload['predictedWasteClass'], payload['actualWasteClass'], payload['createdAt'], payload['imageUrl'], payload['geoTag'], payload['xonecoor'], payload['yonecoor'],payload['xtwocoor'], payload['ytwocoor'])
}



headers={'x-api-key':api_token}
r=requests.post(url=url,json=json,headers=headers)
