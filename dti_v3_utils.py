import re
import sys
import pickle
import os
import json
import asyncio
import sched
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import traceback
import numpy as np
from clickhouse_driver.client import Client
from clickhouse_driver.errors import ServerException, SocketTimeoutError
from concurrent.futures import ThreadPoolExecutor
import pymysql
import logging
import logging.config
from logging.handlers import QueueHandler, QueueListener
import queue
import datetime

'''load pwd'''
pwd = sys.path[0]

'''load config json'''
with open(pwd + '/conf/config.json') as f:
    config = json.loads(json.dumps(json.load(f)))

    
def train_test_split(indexes, data, labels, split=0.7):
    permutation = np.random.permutation(len(indexes))
    train_size = int(len(indexes)*0.7)
    data_dict = {}
    data_dict['IDX_train'] = indexes.values[permutation[:train_size]]
    data_dict['IDX_test'] = indexes.values[permutation[train_size:]]
    data_dict['X_train'] = data.values[permutation[:train_size]]
    data_dict['Y_train'] = labels[permutation[:train_size]]
    data_dict['X_test'] = data.values[permutation[train_size:]]
    data_dict['Y_test'] = labels[permutation[train_size:]]
    data_dict['X_train'] = data_dict['X_train'].reshape(data_dict['X_train'].shape[0], data_dict['X_train'].shape[1], 1)
    data_dict['X_test'] = data_dict['X_test'].reshape(data_dict['X_test'].shape[0], data_dict['X_test'].shape[1], 1)
    return data_dict


def check_cs(index=0):
    cs = config['cs']
    if index >= len(cs):
        logging.error('[clickhouse client ERROR] connect fail')
        return None
    
    '''입력 받은 config index 위치 출력'''
    ch = cs[index]
    print(ch)
    
    try:
        client = Client(ch['host'], port=ch['port'],
                        send_receive_timeout=int(ch['timeout']),
                        settings={'max_threads': int(ch['thread'])}
                       )
        client.connection.force_connect()
        if client.connection.connected:
            return client
        else:
            return check_cs(index + 1)
    except:
        return check_cs(index + 1)

def execute_ch(sql, param=None, with_column_types=True):
    client = check_cs(index=0)
    print(client)
    if client == None:
        sys.exit(1)

    result = client.execute(sql, params=param, with_column_types=with_column_types)

    client.disconnect()
    return result
    
    # def get_index(self):
    #     return self.index

async def execute_async_ch(sql, param=None):
    client = check_async_cs(0)
    if client == None:
        sys.exit(1)
    
    result = await client.execute(sql, param)

    client.disconnect()
    return result

def check_async_cs(index):
    cs = config['cs']
    if index >= len(cs):
        logger.error('[clickhouse async client ERROR] connect fail')
        return None
    ch = cs[index]

    try:
        client = AsyncClient(ch["host"], port=ch["port"], send_receive_timeout=int(ch['timeout']), settings={"max_threads":int(ch['thread'])})
        client.connection.force_connect()
        if client.connection.connected:
            logger.info('[clickhouse async client.execute(sql)] connected to {}'.format(ch))
            return client
        else:
            return check_async_cs(index)
#             return check_async_cs(index+1)
    except:
        return check_async_cs(index)
#         return check_async_cs(index+1)

async def _run_in_executor(executor, func, *args, **kwargs):
    if kwargs:
        func = partial(func, **kwargs)
    loop = asyncio.get_event_loop()

    return await loop.run_in_executor(executor, func, *args)


class AsyncClient(Client):
    def __init__(self, *args, **kwargs):
        self.executor = ThreadPoolExecutor(max_workers = 1)
        super(AsyncClient, self).__init__(*args, **kwargs)

    async def execute(self, *args, **kwargs):
        return await _run_in_executor(self.executor, super(AsyncClient, self).execute, *args, **kwargs)
    
    
class Params():
    """
    load config logging
    """
    if not os.path.exists(pwd+'/logs'):
        os.makedirs(pwd+'/logs') 

    def set_logger(mode, model_id, defult=False):
        """
        pwd(지정한 경로)에 log 파일 생성
        """
        if defult:
            logconf = json.loads(json.dumps(config['logging']).replace('{DIR}', pwd).replace('{MODE}', 'default').replace('{MODEL_ID}', 'ai'))
        else:
            logconf = json.loads(json.dumps(config['logging']).replace('{DIR}', pwd).replace('{MODE}', mode).replace('{MODEL_ID}', str(model_id)))
        logging.config.dictConfig(logconf)
        logger = logging.getLogger()
        logger.propagate = False

        #log queue
        q = queue.Queue(-1) #unlimit
        q_handler = QueueHandler(q)
        q_listener = QueueListener(q, logger.handlers)
        q_listener.start()
        logger.info('logging config : {}'.format(logconf))
        return logger
    
    globals()['logger'] = set_logger(None, None, True)

    def _get_delta(delta):
        """
        load mySQL config (DTI 내부 AI 모델 설정 config) & now_delta, prev_delta 시간 변환
        """
        try:
            delta = delta.strip()
            delta = delta.replace(' ', '')
            unit, num = delta.split('=')[0], int(delta.split('=')[1])
            if unit == 'seconds':
                return datetime.timedelta(seconds=num)
            elif unit == 'minutes':
                return datetime.timedelta(minutes=num)
            elif unit == 'hours':
                return datetime.timedelta(hours=num)
            elif unit == 'days':
                return datetime.timedelta(days=num)
            elif unit == 'weeks':
                return datetime.timedelta(weeks=num)
            else:
                logger.error('[getDelta ELSE] delta: {} => return default value: days=1'.format(delta))
                return datetime.timedelta(days=1)
        except:
            logger.error('[getDelta ERROR] delta: {} => return default value: days=1'.format(delta))
            return datetime.timedelta(days=1)
     
    @classmethod 
    def get_config(cls, mode, model_id, model_name, TRAIN=True):
        """
        mySQL config load
        """
        conn = pymysql.connect(host = config['mysql']['host'], port = config['mysql']['port'], user = config['mysql']['user'], password = config['mysql']['password'], db = config['mysql']['db'])
        curs = conn.cursor()
        sql = 'select config from model_meta where model_id = {}'.format(model_id)
        curs.execute(sql)
        result = list(curs.fetchone())[0]
        
        if mode == 'TRAIN':
            model_config = json.loads(result)['train']
        elif mode == 'TEST_TRAIN':
            model_config = json.loads(result)['train']
        elif mode == 'PREDICTION':
            model_config = json.loads(result)['predict']
        elif mode == 'TEST_PREDICTION':
            model_config = json.loads(result)['predict']
        elif mode == 'COMMON':
            model_config = json.loads(result)['common']
        else:
            model_config = json.loads(result)
        conn.close()
        
        mode_list = ['TRAIN', 'TEST_TRAIN', 'PREDICTION', 'TEST_PREDICTION']
        
        for i in mode_list:
            if mode == i:
                model_config["common"] = json.loads(result)['common']
                model_config['now_delta'] = cls._get_delta(model_config['now_delta'])
                model_config['prev_delta'] = cls._get_delta(model_config['prev_delta'])
            
        logger.info('[{}({})] model config : {}'.format(mode, model_name, model_config))
        return model_config


class Object():
    """
    save/load json & obj
    """
    if not os.path.exists(pwd+'/obj'):
        os.makedirs(pwd+'/obj') 
    
    def save_json(file, name='model_config.json'):
        try:
            with open(pwd+'/config/'+name, 'w') as outfile:
                json.dump(file, outfile)
            print('success! save_json', name)
        except:
            print('Fail! save_json')

    def load_json(name):
        try:
            with open(pwd+'/config/'+name) as json_file:
                model_config = json.loads(json_file.read())
            print('success! load_json', name)
            return model_config
        except:
            print('Fail! load_json')

    def save_obj(obj, name):
        try:
            with open(pwd+'/obj/{}.pickle'.format(name), 'wb') as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
            print('success! save_obj: {}.pickle'.format(name))
        except:
            print('Fail! save_obj')

    def load_obj(name):
        try:
            with open(pwd+'/obj/{}.pickle'.format(name), 'rb') as f:
                print('success! load_obj: {}.pickle'.format(name))
                return pickle.load(f)
        except:
            print('Fail! load_obj')

    def update_obj(name):
        try:
            from_obj = '{}/obj/{}.pickle'.format(pwd, name)
            to_obj = '{}/obj/{}.pickle'.format(pwd, name)

            if os.path.exists(to_obj):
                os.system('rm -rf {}'.format(to_obj))

            os.system('mv {} {}'.format(from_obj, to_obj))
            logger.info('update_obj: {} -> {}'.format(from_obj, to_obj))
        except:
            print('Fail! update_obj')

            
'''Constant Parameter'''
# Click House 에서 사용되는 Column 의 Type 값 모음
CLICK_HOUSE_TYPES = ['Int8', 'Int16', 'Int32', 'Int64', 'Int128', 'Int256',
                     'UInt8', 'UInt16', 'UInt32', 'UInt64', 'UInt128', 'UInt256',
                     'Float32', 'Float64', 'Enum8', 'Enum16',
                     'Date', 'DateTime', 'DateTime64', 'IPv4', 'IPv6',
                     'String', 'FixedString', 'UUID', 'Decimal', 'Nested'
                     ]
# Click House 에서 사용되는 Column 의 Type 값 중 다른 값을 포함할 수 있는 값 모음
CLICK_HOUSE_WRAPPED_TYPES = ['Tuple', 'Array', 'Nullable', 'LowCardinality', 'Map']

# Click House 에서 지원하는 테이블 수정 명령어 모음
MODIFY_TYPE = ['ADD', 'DROP', 'RENAME', 'CLEAR', 'COMMENT', 'MODIFY']

# SQL
CONNECTION_TEST_SQL = 'SELECT 1'
DROP_TABLE_SQL = 'DROP TABLE IF EXISTS %s'
TABLE_CHECK_SQL = 'SELECT 1 FROM %s'
ALTER_TABLE_SQL = 'ALTER TABLE %s %s COLUMN '

# 안내 메세지
DB_CONNECTION_FAIL = '데이터베이스 연결에 실패했습니다.'
NAME_ERROR = '데이터베이스 혹은 테이블 이름 유효성 검사에 실패했습니다.'
COLUMN_NAME_OR_TYPE_ERROR = 'Column 정보가 유효하지 않습니다. Column 이름과 타입을 확인해주세요.'
TABLE_CREATE_SUCCESS = '테이블 생성에 성공했습니다.'
TABLE_DELETE_SUCCESS = '테이블 삭제에 성공했습니다.'
TABLE_CREATE_FAIL = '테이블 생성에 실패했습니다.'
TABLE_DELETE_FAIL = '테이블 삭제에 실패했습니다.'
TABLE_ALREADY_EXISTS = '테이블이 이미 존재합니다.'
TABLE_DOESNT_EXISTS = '테이블이 존재하지 않습니다.'
TABLE_MODIFY_SUCCESS = '테이블 수정에 성공했습니다.'
TABLE_MODIFY_FAIL = '테이블 수정에 실패했습니다.'
MODIFY_TYPE_ERROR = '테이블 수정을 위한 명령어가 유효하지 않습니다. 가능한 명령어: ADD, DROP, RENAME, CLEAR, COMMENT, MODIFY'


def column_name_check(columns):
    """
    Table 생성할 때 Column 이름이나 Type 에 대한 유효성 검사를 위한 메소드
    :param columns: 튜플이나 리스트 형태의 데이터로 (Column 이름, Type) 쌍의 데이터의 모음
    :return: 유효성 검사 결과(Boolean)
    """
    for c in columns:
        if not regex_check_eng_num_u(c[0]):
            return False
        if c[1] not in CLICK_HOUSE_TYPES and c[1].split('(')[0] not in CLICK_HOUSE_WRAPPED_TYPES:
            return False
    return True


def regex_check_eng_num_u(string):
    """
    영어나 숫자, _(underbar)로만 이루어진 표현인지 검증하기 위한 메소드
    :param string: 검증 대상 문자열
    :return: 검증 결과(Boolean)
    """
    regex_eng_num_u = re.compile(r'[a-zA-Z0-9_]')
    if regex_eng_num_u.match(string):
        return True
    return False


class DBCheck:
    """
Click House 데이터베이스에 연결하고 테이블을 생성하거나 삭제, 수정하기 위한 클래스
데이터베이스의 각 테이블 마다 객체를 생성하여 사용한다.
    """
    
    def __init__(self, index, database_name: str, table_name: str):
        """
        :param database_name: 데이터베이스의 이름, 유효성 검사를 통과해야 한다.
        :param table_name: 테이블의 이름, 유효성 검사를 통과해야 한다.
        :return:
        """
        self.index = index

        cs = config['cs']
        if index >= len(cs):
            logging.error('[clickhouse client ERROR] connect fail')
            return None
        ch = cs[index]
        
        if not regex_check_eng_num_u(database_name) or not regex_check_eng_num_u(table_name):
            print(NAME_ERROR)
            return
        self.table_name = database_name + '.' + table_name
        self._client = Client(ch['host'], port=ch['port'])
        try:
            self._client.execute(CONNECTION_TEST_SQL)
        except SocketTimeoutError as ste:
            print(DB_CONNECTION_FAIL)
            return

    def check_table_exist(self):
        """
        테이블이 존재하는지 검사
        :return: 테이블 존재 여부(Boolean)
        """
        try:
            sql = TABLE_CHECK_SQL % self.table_name
            self._client.execute(sql)
        except ServerException as se:
            return False
        return True

    def create_table(self, columns, engine='MergeTree', partition='tuple()', order='tuple()'):
        """
        테이블이 존재하지 않을 경우 테이블을 생성한다.
        :param columns: 테이블의 Column 목록, 유효성 검사를 통과해야 한다.
        :param engine: Click House 의 테이블에 사용할 엔진을 지정한다. Def: MergeTree
        :param partition: 파티션의 기준을 지정한다. Def: tuple()
        :param order: 테이블의 정렬 순서를 지정한다. Def: tuple()
        :return: 테이블 생성 결과(Boolean)
        """
        if not column_name_check(columns):
            print(COLUMN_NAME_OR_TYPE_ERROR)
            return False
        if self.check_table_exist():
            print(TABLE_ALREADY_EXISTS)
            return False
        try:
            sql = 'CREATE TABLE ' + self.table_name \
                  + '(' + ','.join(['`' + str(c[0]) + '` ' + str(c[1]) for c in columns]) + ')'\
                  + ' ENGINE = ' + engine\
                  + ' PARTITION BY ' + partition\
                  + ' ORDER BY ' + order\
                  + ' SETTINGS index_granularity = 8192'
            self._client.execute(sql)
            print(TABLE_CREATE_SUCCESS)
            return True
        except ServerException as se:
            print(se.message)
            print(TABLE_CREATE_FAIL)
            return False

    def delete_table(self):
        """
        테이블이 존재하는 경우 테이블을 삭제한다.
        :return: 테이블 삭제 결과(Boolean)
        """
        if not self.check_table_exist():
            print(TABLE_DOESNT_EXISTS)
        print('테이블이 삭제됩니다. 정말 삭제하시려면 테이블 이름을 입력하세요. (데이터베이스명 포함)')
        name = input()
        if name != self.table_name:
            print('잘못 입력하셨습니다. 삭제가 취소됩니다.')
            return
        try:
            sql = DROP_TABLE_SQL % self.table_name
            self._client.execute(sql)
            print(TABLE_DELETE_SUCCESS)
            return True
        except ServerException as se:
            print(se.message)
            print(TABLE_DELETE_FAIL)
            return False

    def modify_table(self, modify_type, params):
        """
        :param modify_type: ADD, DROP, RENAME, CLEAR, COMMENT, MODIFY
        :param params: 각 Type 에 맞는 파라미터 값, String 형태
        ADD 의 경우 column_name column_type options(FIRST, AFTER column)
        DROP 의 경우 column_name
        :return: 테이블 수정 결과(Boolean)
        """
        if modify_type not in MODIFY_TYPE:
            print(MODIFY_TYPE_ERROR)
            return False
        try:
            sql = ALTER_TABLE_SQL % (self.table_name, modify_type) + params
            self._client.execute(sql)
            print(TABLE_MODIFY_SUCCESS)
            return True
        except ServerException as se:
            print(se.message)
            print(TABLE_MODIFY_FAIL)
            return False            
                        


def isExistModel(model_id):
    conn = pymysql.connect(host = config['mysql']['host'], port = config['mysql']['port'], user = config['mysql']['user'], password = config['mysql']['password'], db = config['mysql']['db'])
    curs = conn.cursor()
    sql = 'select model_name from model_meta where model_id = {}'.format(model_id)
    result = curs.execute(sql)

    if result == 0:
        model_name = None
    else:
        model_name = list(curs.fetchone())[0]

    conn.close()
    return model_name


def getToStartOf(crontab):
    """
    Scheduler Time Setting
    """
    try:
        m, h, d, M, y = crontab.split(' ')
        if m == '*/1' or crontab == '* * * * *':
            return 'toStartOfMinute'
        elif m == '*/5':
            return 'toStartOfFiveMinute'
        elif m == '*/15':
            return 'toStartOfFifteenMinutes'
        elif h == '*/1':
            return 'toStartOfHour'
        elif d == '*/1':
            return 'toStartOfDay'
        elif d == '*/4':
            return 'toStartOfQuarter'
        elif M == '*/1':
            return 'toStartOfMonth'
        elif y == '*/1':
            return 'toStartOfYear'
        else:
            logger.info('[getToStartOf ELSE] crontab: {} => return default value: toStartOfHour'.format(crontab))
            return 'toStartOfHour'
    except:
        logger.error('[getToStartOf ERROR] crontab: {} => return default value: toStartOfHour'.format(crontab))
        return 'toStartOfHour'
    
            
class BaseComponent(Object):
    """
    Run Scheduler
    """
    def __init__(self, loop, model_id, model_name):
        self.loop = loop
        self.model_id = model_id
        self.model_name = model_name
        self.update_topic = 'update_{}'.format(model_id)
        self.model = None
        self.http_doc_dict = None

        if not os.path.exists(pwd+'/logs/'):
            logger.info('create directory: {}'.format(pwd+'/logs/'))
            os.makedirs(pwd+'/logs/')
        if not os.path.exists(pwd+'/graph/'):
            logger.info('create directory: {}'.format(pwd+'/graph/'))
            os.makedirs(pwd+'/graph/')
        if not os.path.exists(pwd+'/check/'):
            logger.info('create directory: {}'.format(pwd+'/check/'))
            os.makedirs(pwd+'/check/')
        if not os.path.exists(pwd+'/obj/'):
            logger.info('create directory: {}'.format(pwd+'/obj/'))
            os.makedirs(pwd+'/obj/')

    def init(self):
        pass

    async def run(self, param):
        pass

    async def update(self, msg):
        model_id = msg.data.decode()
        self.model.load()
        logger.info('[UPDATE] model update: {}'.format(model_id))
        
    async def run_scheduler(self, mode):
        try:
            self.logger = Params.set_logger(mode, self.model_id)
            now = datetime.datetime.now() - self.model_config['now_delta'] + datetime.timedelta(hours=9)
            prev = now - self.model_config['prev_delta']
            prev_month = now - datetime.timedelta(weeks=8)
            prev_day = now - datetime.timedelta(days=1)
            logger.info('[{}({})] {} ~ {}'.format(mode, self.model_name, prev, now))
            
            param = {
                'logdate_s': prev.strftime('%Y-%m-%d'),
                'logdate_e': now.strftime('%Y-%m-%d'),
                'logtime_s': prev.strftime('%Y-%m-%d %H:%M:%S'),
                'logtime_e': now.strftime('%Y-%m-%d %H:%M:%S'),
                'logdate_m': prev_month.strftime('%Y-%m-%d'),
                'logdate_day': prev_day.strftime('%Y-%m-%d'),
                'result_table': self.model_config['result_table'],
#                 'cluster_table': self.model_config['cluster_table'],
#                 'model_hist': self.model_config['history_table'],
                'toStartOf': getToStartOf(self.model_config['crontab'])
            }
            
            logger.info('[{}({})] param: {}'.format(mode, self.model_name, param))
            task = [self.run(param)]
            
            for f in asyncio.as_completed(task):
                task_result = await f
                logger.info('[{}({})] result: {}'.format(mode, self.model_name, task_result))
                if task_result == None:
                    sys.exit(1)
                    
        except Exception as err:
            logger.error(err)
            logger.error(traceback.print_exc())

    async def start_train(self):
        try:
            self.model_config = Params.get_config('TRAIN', self.model_id, self.model_name)
            scheduler = AsyncIOScheduler()
            scheduler.add_job(self.run_scheduler, CronTrigger.from_crontab(self.model_config['crontab']), ['TRAIN'], misfire_grace_time=3600, max_instances=20)
            scheduler.start()
            
        except Exception as err:
            logger.error(err)
            logger.error(traceback.print_exc())
            
    async def test_train(self):
        try:
            self.model_config = Params.get_config('TEST_TRAIN', self.model_id, self.model_name)
            await self.run_scheduler('TEST_TRAIN')
            
        except Exception as err:
            logger.error(err)
            logger.error(traceback.print_exc())
            
    async def start_pred(self):
        try:
            self.model_config = Params.get_config('PREDICTION', self.model_id, self.model_name, False)
            scheduler = AsyncIOScheduler()
            scheduler.add_job(self.run_scheduler, CronTrigger.from_crontab(self.model_config['crontab']), ['PREDICTION'], misfire_grace_time=3600, max_instances=20)
            scheduler.start()
            
        except Exception as err:
            logger.error(err)
            logger.error(traceback.print_exc())
            
    async def test_pred(self):
        try:
            self.model_config = Params.get_config('TEST_PREDICTION', self.model_id, self.model_name, False)
            await self.run_scheduler('TEST_PREDICTION')
            
        except Exception as err:
            logger.error(err)
            logger.error(traceback.print_exc())
            

            



