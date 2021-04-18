from binance.client import Client
import config
import pandas as pd
import dateparser
import pytz
# from datetime import timezone

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

  def get_klines_pd(self, symbol, interval=Client.KLINE_INTERVAL_1MINUTE,
                    limit=1000, tz='Asia/Shanghai'):
    candles = self.client.get_klines( symbol=symbol,
                                      interval=interval,
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

  def _to_utc_datetime_str(self, local_datetime_str, tz):
    #utc_dt = pd.to_datetime(local_datetime_str).tz_localize(tz).tz_convert('UTC')
    utc_dt = dateparser.parse(local_datetime_str).astimezone(pytz.utc)
    return utc_dt.strftime('%Y-%m-%d %H:%M:%S')

  def get_historical_klines_pd(self, symbol, start_str, end_str=None,
                               interval=Client.KLINE_INTERVAL_1MINUTE,
                               limit=1000, tz='Asia/Shanghai'):
    start_t = self._to_utc_datetime_str(start_str, tz)
    end_t = None if end_str is None else self._to_utc_datetime_str(end_str, tz)

    candles = self.client.get_historical_klines(symbol=symbol,
                                                interval=interval,
                                                start_str=start_t,
                                                end_str=end_t,
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