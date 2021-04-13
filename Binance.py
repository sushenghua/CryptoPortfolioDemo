from binance.client import Client
import config
import pandas as pd

class BinanceClient:
  __instance = None

  def instance():
    if BinanceClient.__instance == None:
      BinanceClient()
    return BinanceClient.__instance

  def __init__(self):
    if BinanceClient.__instance == None:
      self.client = Client(config.api_key,
                           config.api_sec,
                           requests_params=config.params,
                           tld=config.tld)
      BinanceClient.__instance = self
    else:
      raise Exception('try to initialize multiple instance of a singleton')

  def get_kline_pd(self, symbol, limit=1000, tz='Asia/Shanghai'):
    candles = self.client.get_klines( symbol=symbol,
                                      interval=Client.KLINE_INTERVAL_1MINUTE,
                                      limit=limit)
    df = pd.DataFrame(candles, dtype=float, columns = ( 'OpenTime',
                                                        'Open',
                                                        'High',
                                                        'Low',
                                                        'Close',
                                                        'Volume',
                                                        'CloseTime',
                                                        'QuoteAssetVolume',
                                                        'NumberOfTrades',
                                                        'TakerBuyBaseAssetVolume',
                                                        'TakerBuyQuoteAssetVolume',
                                                        'Ignore' ))
    df['OpenTime'] = pd.to_datetime(df['OpenTime'], unit='ms', utc=True).dt.tz_convert(tz)
    df['CloseTime'] = pd.to_datetime(df['CloseTime'], unit='ms', utc=True).dt.tz_convert(tz)
    return df

  def call_method(f, **p):
    ret = None
    if f is not None and hasattr(self.client, f):
      try:
        if p is not None:
          ret = getattr(self.client, f)(**p)
        else:
          ret = getattr(self.client, f)()
      except Exception as e:
        raise Exception('binance function error: {}'.format(e))
    else:
      raise Exception('binance function not found')
    return ret