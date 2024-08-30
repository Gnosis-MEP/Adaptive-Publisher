import uuid

import snappy

from event_service_utils.img_serialization.redis import RedisImageCache


class CompressKeyRedisImageCache(RedisImageCache):


    def upload_inmemory_to_storage_compress(self, img_numpy_array):
        img_key = str(uuid.uuid4())
        nd_array_bytes = img_numpy_array.tobytes(order='C')
        compress = snappy.compress(nd_array_bytes)
        ret = self.client.set(img_key, compress)
        if ret:
            self.client.expire(img_key, self.expiration_time)
        else:
            raise Exception('Couldnt set image in redis')
        return img_key