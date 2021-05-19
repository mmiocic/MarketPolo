import json
import requests
import os
import pprint
import uuid

session = requests.Session()
account_url = None

VALID_TIMES = {'5minute', '10minute', 'hour', 'day', 'week', 'month', '3month', 'year', 'all'}
VALID_BOUNDS = {'regular', 'trading', 'extended', '24_7'}

PAIRS = {
    'BTCUSD': '3d961844-d360-45fc-989b-f6fca761d511',
    'ETHUSD': '76637d50-c702-4ed1-bcb5-5b0732a81f48',
    'ETCUSD': '7b577ce3-489d-4269-9408-796a0d1abb3a',
    'BCHUSD': '2f2b77c4-e426-4271-ae49-18d5cb296d3a',
    'BSVUSD': '086a8f9f-6c39-43fa-ac9f-57952f4a1ba6',
    'LTCUSD': '383280b1-ff53-43fc-9c84-f01afd0989cd',
    'DOGEUSD': '1ef78e1b-049b-4f12-90e5-555dcf2fe204'
}

ENDPOINTS = {
    'auth': 'https://api.robinhood.com/oauth2/token/',
    'currency_pairs': 'nummus.robinhood.com/currency_pairs',
    'quotes': 'https://api.robinhood.com/marketdata/forex/quotes/{}/',
    'historicals': 'https://api.robinhood.com/marketdata/forex/historicals/{}/?interval={}&span={}&bounds={}',
    'orders': 'https://nummus.robinhood.com/orders/',
    'order_status': 'https://nummus.robinhood.com/orders/{}',  # Order id
    'order_cancel': 'https://nummus.robinhood.com/orders/{}/cancel/',
    'nummus_accounts': 'https://nummus.robinhood.com/accounts/',
    'holdings': 'https://nummus.robinhood.com/holdings/',
    'api_accounts': 'https://api.robinhood.com/accounts/',
    'portfolios': 'https://api.robinhood.com/accounts/{}/portfolio/'
}


def oauth(payload):
    global session

    url = 'https://api.robinhood.com/oauth2/token/'
    r = session.post(url, json=payload)
    if r.status_code == 500:
        raise RuntimeError('Missing or incorrect credentials.')
    session.headers.pop('challenge_id', None)
    response = r.json()

    if 'access_token' in response:
        # save bearer_token and write to tokens.json
        session.headers['Authorization'] = 'Bearer ' + response['access_token']
        with open('tokens.json', 'w') as file:
            file.write(json.dumps({
                'bearer_token': response['access_token'],
                'refresh_token': response['refresh_token'],
                'device_token': payload['device_token']
            }))

    return r


def login(username: str = None, password: str = None, mfa_code: str = None,
          device_token: str = 'c77a7142-cc14-4bc0-a0ea-bdc9a2bf6e68',
          bearer_token: str = None, no_input: bool = False) -> str:
    """generates and returns OAuth2 bearer token"""
    global session

    if bearer_token is not None:
        session.headers['Authorization'] = 'Bearer ' + bearer_token
        if is_logged_in():
            return bearer_token
        else:
            print('Invalid/expired bearer token')
            del session.headers['Authorization']
    # check if bearer token exists and is valid. create tokens.json if does not exist.
    if os.path.isfile('tokens.json'):
        with open('tokens.json', 'r') as file:
            try:
                tokens = json.loads(file.read())
                if 'bearer_token' in tokens:
                    bearer_token = tokens['bearer_token']
                    session.headers['Authorization'] = 'Bearer ' + bearer_token
                    if is_logged_in():
                        return bearer_token
                    else:
                        del session.headers['Authorization']
            except json.decoder.JSONDecodeError:
                pass

    if username is None and not no_input:
        username = input('Enter email or username: ')
    if password is None and not no_input:
        password = input('Enter password: ')

    payload = {
        'grant_type': 'password',
        'client_id': 'c82SH0WZOsabOXGP2sxqcj34FxkvfnWRZBKlBjFS',
        'device_token': device_token,
        'username': username,
        'password': password
    }
    if mfa_code is not None:
        payload['mfa_code'] = mfa_code

    r = oauth(payload)
    if r.status_code == 400:
        r = r.json()
        if 'detail' in r and r['detail'] == 'Request blocked, challenge type required.':
            challenge_type = None
            while challenge_type not in ['1', '2']:
                print('Unfamiliar device detected.')
                challenge_type = '1' if no_input else \
                    input("We're sending you a code to verify your login. Do you want us to:\n"
                          "  1: Text you the code\n"
                          "  2: Email it to you?\n")
                if challenge_type == '1':
                    print('Texting...')
                    payload['challenge_type'] = 'sms'
                elif challenge_type == '2':
                    print('Emailing...')
                    payload['challenge_type'] = 'email'
            r = oauth(payload)
            del payload['challenge_type']
            challenge_id = r.json()['challenge']['id']
            url = f'https://api.robinhood.com/challenge/{challenge_id}/respond/'
            verified = False
            while verified is False:
                verification_code = input('\nEnter your verification code: ')
                r = session.post(url, json={'response': verification_code}).json()
                if 'id' in r:
                    verified = True
                    print('\nVerified device.\n')
                else:
                    remaining_attempts = r['challenge']['remaining_attempts']
                    if remaining_attempts > 0:
                        print(f"Code is invalid. Remaining attempts: {remaining_attempts}.")
                    else:
                        raise RuntimeError('Verification failed.')
            session.headers['X-ROBINHOOD-CHALLENGE-RESPONSE-ID'] = challenge_id
            oauth(payload)
            del session.headers['X-ROBINHOOD-CHALLENGE-RESPONSE-ID']
        else:
            raise RuntimeError('Unable to log in with provided credentials.')

    elif r.status_code == 401:
        raise RuntimeError('Invalid bearer token.')
    r = r.json()
    if r.get('mfa_required'):
        if no_input:
            raise RuntimeError('Multi-factor authentication is enabled. "mfa_code" required.')
        else:
            mfa_code = input('Enter the code generated by your authentication app: ')
            return login(username=username, password=password, mfa_code=mfa_code, device_token=device_token,
                         bearer_token=bearer_token, no_input=no_input)
    else:
        return r['access_token']


def is_logged_in() -> bool:
    """checks whether user is logged in"""
    url = "https://api.robinhood.com/user/"
    r = session.get(url)
    if r.status_code == 401:
        # invalid bearer token
        return False
    else:
        print(f"Logged in as {r.json()['profile_name']}.\n")
        return True


def user() -> dict:
    url = "https://api.robinhood.com/user/"
    r = session.get(url)
    return r.json()


def accounts() -> dict:
    global account_url
    url = 'https://api.robinhood.com/accounts/'
    r = session.get(url).json()
    account_url = r['results'][0]['url']
    return r['results'][0]


def instruments(instrument=None, symbol=None) -> dict:
    url = 'https://api.robinhood.com/instruments/'
    if instrument is not None:
        url += f'{instrument}/'
    if symbol is not None:
        url += f'?symbol={symbol}'
    r = session.get(url)
    return r.json()


def positions(nonzero: bool = True) -> dict:
    url = 'https://api.robinhood.com/positions/'
    r = session.get(url, params={'nonzero': str(nonzero).lower()})
    if r.status_code == 401:
        raise RuntimeError(
            r.text + '\nYour bearer_token may have expired.')
    r = r.json()
    positions = {}
    for result in r['results']:
        instrument_url = result['instrument']
        r = session.get(instrument_url).json()
        positions[r['symbol']] = {
            'quantity': result['quantity'],
            'average_buy_price': result['average_buy_price']
        }
    return positions


def options_positions(nonzero: bool = True) -> dict:
    url = 'https://api.robinhood.com/options/aggregate_positions/'
    r = session.get(url, params={'nonzero': nonzero})
    return r.json()


def live(account_number, span: str = 'day') -> dict:
    if span not in VALID_TIMES:
        raise RuntimeError(f"'{span}' is not valid as span.")
    url = 'https://api.robinhood.com/historical/portfolio_v2/live/'
    r = session.get(url, params={
        'account_number': account_number,
        'span': span,
        'from': 0
    })
    return r.json()


def fundamentals(instrument) -> dict:
    url = f'https://api.robinhood.com/fundamentals/{instrument.upper()}/'
    r = session.get(url)
    return r.json()


def quotes(instrument) -> dict:
    url = f'https://api.robinhood.com/marketdata/quotes/{instrument.upper()}/'
    r = session.get(url)
    return r.json()


def c_quotes(instrument) -> dict:
    url = f'https://api.robinhood.com/marketdata/forex/quotes/{instrument.upper()}/'
    r = session.get(url)
    return r.json()


def historicals(instrument: str, bounds: str = 'regular', interval: str = '5minute', span: str = 'day') -> dict:
    if bounds not in VALID_BOUNDS:
        raise RuntimeError(f"'{bounds}' is not valid as bounds.")
    if interval not in VALID_TIMES:
        raise RuntimeError(f"'{interval}' is not valid as interval.")
    if span not in VALID_TIMES:
        raise RuntimeError(f"'{span}' is not valid as span.")
    url = f'https://api.robinhood.com/marketdata/historicals/{instrument}/?bounds={bounds}&interval={interval}'
    r = session.get(url).json()
    return r


def orders(price, symbol, instrument=None, quantity=1, type='market', side='buy', time_in_force='gfd',
           trigger='immediate', account=None) -> dict:
    global account_url
    if account is None:
        account = account_url
    if instrument is None:
        instrument = fundamentals(symbol)['instrument']
    url = 'https://api.robinhood.com/orders/'
    r = session.post(url, json=locals())
    return r.json()


def search(query: str):
    url = 'https://api.robinhood.com/midlands/search/'
    r = session.get(url, params={'query': query})
    return r.json()


def crypto_positions(nonzero: bool = True) -> dict:
    # https://github.com/wang-ye/robinhood-crypto
    # go here to complete the crypto side of RH

    url = 'https://nummus.robinhood.com/holdings/'
    r = session.get(url, params={'nonzero': nonzero})
    r = r.json()
    crypto_positions = {}
    for result in r['results']:
        cum_quant = 0
        cum_cost = 0
        for cb in result['cost_bases']:
            cum_quant += float(cb['direct_quantity'])
            cum_cost += float(cb['direct_cost_basis'])
        crypto_positions[result['currency']['code']] = {
            'total_quantity': cum_quant,
            'total_cost': cum_cost
        }
    return crypto_positions


def c_accounts():
    # url = ENDPOINTS['nummus_accounts']
    # try:
    #     data = session.get(url)
    # except Exception as e:
    #     raise e
    # if 'results' in data:
    #     return [x for x in data['results']]
    # return []
    url = ENDPOINTS['nummus_accounts']
    r = session.get(url)
    r = r.json()
    # account_url = r['results'][0]['url']
    print(r['results'])
    return r['results'][0]


# def accounts() -> dict:
#     global account_url
#     url = 'https://api.robinhood.com/accounts/'
#     r = session.get(url).json()
#     account_url = r['results'][0]['url']
#     return r['results'][0]


def crypto_account_id() -> dict:
    accounts_info = c_accounts()
    pprint.pprint(accounts_info)
    if accounts_info:
        return accounts_info['id']
    else:
        # LOG.error('account cannot be retrieved')
        # raise AccountNotFoundException()
        print('Error - account cannot be retrieved')
    return None


def trade(pair, **kwargs):
    assert pair in PAIRS.keys(), 'pair {} is not in {}.'.format(pair, PAIRS.keys())
    set(kwargs.keys()) == ['price', 'quantity', 'side', 'time_in_force', 'type']
    payload = {
        **{
            'account_id': crypto_account_id(),
            'currency_pair_id': PAIRS[pair],
            'ref_id': str(uuid.uuid4()),
        },
        **kwargs
    }
    try:
        # res = session('https://nummus.robinhood.com/orders/', json_payload=payload, method='post', timeout=5)
        # res = session.post('https://nummus.robinhood.com/orders/', payload)
        res = session.request(method='post',
                              url='https://nummus.robinhood.com/orders/',
                              json=payload,
                              headers={'content-type': 'application/json'})
    except Exception as e:
        # raise TradeException()
        print(e)
    print(res.json())
    return res.json()


# return value:
# {
# 'account_id': 'abcd', 'cancel_url': None, 'created_at': '2018-04-22T14:07:37.103809-04:00', 'cumulative_quantity': '0.000111860000000000', 'currency_pair_id': '3d961844-d360-45fc-989b-f6fca761d511', 'executions': [{'effective_price': '8948.500000000000000000', 'id': 'hijk', 'quantity': '0.000111860000000000', 'timestamp': '2018-04-22T14:07:37.329000-04:00'}], 'id': 'order_id', 'last_transaction_at': '2018-04-22T14:07:37.329000-04:00', 'price': '9028.670000000000000000', 'quantity': '0.000111860000000000', 'ref_id': 'ref_id', 'side': 'buy', 'state': 'filled', 'time_in_force': 'gtc', 'type': 'market', 'updated_at': '2018-04-22T14:07:38.956584-04:00'
# }
def order_status(order_id):
    url = ENDPOINTS['order_status'].format(order_id)
    try:
        res = session.get(url)
    except Exception as e:
        raise e
    return res


def order_cancel(self, order_id):
    url = ENDPOINTS['order_cancel'].format(order_id)
    try:
        res = self.session_request(url, method='post')
    except Exception as e:
        raise e
    return res


def main():
    pp = pprint.PrettyPrinter(indent=1, width=100, depth=None, stream=None, compact=False, sort_dicts=False)

    # you will be prompted for your username and password
    login()

    print('===User Info===\n')
    pp.pprint(user())
    print('\n')

    print('===Account Info===\n')
    pp.pprint(accounts())
    print('\n')

    print('===Positions===\n')
    pp.pprint(positions())
    print('\n')

    # print('===Options Positions===\n')
    # pprint.pprint(options_positions())
    # print('\n')

    print('===Crypto Positions===\n')
    pprint.pprint(crypto_positions())
    print('\n')

    # print(instruments())
    # print(live())
    # print(quotes('MSFT'))
    # print(fundamentals())
    # print(historicals())
    # print(orders())
    # print(search())

    print(c_quotes('ETHUSD'))

    c_quote = c_quotes('ETHUSD')['mark_price']
    print(c_quote)

    ###THIS TRADE WORKED!!!! DO NOT EXECUTE AGAIN###
    # market_order_info = trade(
    #     'ETHUSD',
    #     price=round(float(c_quote), 2),
    #     quantity="0.002941",
    #     side="sell",
    #     time_in_force="gtc",
    #     type="market"
    # )
    pp.pprint(market_order_info)

    order_id = market_order_info['id']
    print('market order {} status: {}'.format(order_id, order_status(order_id)))


if __name__ == "__main__":
    main()
