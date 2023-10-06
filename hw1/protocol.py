import logging
import socket
import time
from enum import Enum
from logging import debug, info

import numpy as np

# logging.getLogger().setLevel(level=logging.DEBUG)


logging.getLogger().setLevel(level=logging.ERROR)


class UDPBasedProtocol:
    def __init__(self, *, local_addr, remote_addr):
        self.udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.remote_addr = remote_addr
        self.udp_socket.bind(local_addr)

    def sendto(self, data):
        return self.udp_socket.sendto(data, self.remote_addr)

    def recvfrom(self, n):
        msg, addr = self.udp_socket.recvfrom(n)
        return msg


class Flag(Enum):
    Empty = np.uint32(0)
    Start = np.uint32(1)
    Ack = np.uint32(2)
    End = np.uint32(4)
    PackBegin = np.uint32(8)
    LastPack = np.uint32(16)


class MyTCPProtocolBase(UDPBasedProtocol):
    # timeout = 5
    timeout = 0.01
    buffer_size = 1 << 16
    # buffer_size = 1 << 11
    pack_size = 24
    die_cnt = 6
    dup_cnt = 2
    max_data_size = 1 << 13
    # max_data_size = 1 << 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq = np.uint(0)
        self.ack = np.uint(0)
        self.received_buffer = b""
        self.receive_index = 0
        self.connected = False
        self.last_pack_to_send = None
        self.pack = False
        self.udp_socket.settimeout(self.timeout)

    @staticmethod
    def make_package(seq: np.uint, ack: np.uint, flag: np.uint32, data: bytes,
                     length=None) -> bytes:
        return seq.tobytes() + ack.tobytes() + flag.tobytes() + np.uint32(
            len(data) if length is None else length).tobytes() + data

    @staticmethod
    def parse_package(package: bytes):
        if len(package) < 30:
            debug(f"Get pack: {package}")
        else:
            debug(f"Get pack: {package[:30]}...")
        res = np.frombuffer(package, count=2, dtype=np.uint)
        seq = res[0]
        ack = res[1]
        flag, length = np.frombuffer(package, offset=16, count=2, dtype=np.uint32)[0:]
        data = package[MyTCPProtocol.pack_size:]
        if len(data) < 30:
            debug(f"Get seq={seq}, ack={ack}, flag={flag}, len={length} data={data}")
        else:
            debug(f"Get seq={seq}, ack={ack}, flag={flag}, len={length} data={data[:30]}...")
        return seq, ack, flag, length, data

    def make_connection_from_client(self):
        info("Start making connection")
        while True:
            pack = self.make_package(np.uint(0), np.uint(0), Flag.Start.value, b"")
            self.smart_sendto(pack)
            pack = self.recv_msg(self.pack_size)
            seq, ack, flag, _, data = self.parse_package(pack)
            if seq == 1 and ack == 0 and flag == Flag.Start.value | Flag.Ack.value:
                self.connected = True
                self.seq = self.ack = np.uint(1)
                return
            logging.warning(f"{self} get start pack with {seq} and {ack}, but wait 1 and 0")

    def smart_sendto(self, data: bytes):
        if len(data) < 30:
            debug(f"{self} sending {data}")
        else:
            debug(f"{self} sending {data[:30]}...")

        self.sendto(data)
        self.last_pack_to_send = data
        return len(data)

    def send_msg(self, package: bytes):
        assert self.smart_sendto(package) == len(package)
        info("Waiting ack by {}".format(self))
        cnt = 0
        while True:
            resp = self.recv_msg(self.pack_size)
            self.receive_index -= self.pack_size
            _, ack, flag, length, _ = self.parse_package(resp)
            debug("Now index of {} is {}".format(self, self.receive_index))
            if flag == Flag.End.value:
                logging.warning("Ending without ack")
                self.end_connection()
                return len(package)
            if ack == self.seq and Flag.Ack.value & flag != 0:
                info(f"{self} success sent. Ack got.")
                return len(package)
            elif Flag.Ack.value & flag == 0:
                logging.critical(f"Flag mismatched: {flag}")
                raise ValueError
            logging.warning(
                f"{self}: Ack and seq mismatch: {ack} vs {self.seq}. " + "Duplicate"
                if ack < self.seq else "Package reorder")
            cnt += 1
            if cnt == self.dup_cnt:
                info("Sending for duplicating prevent")
                self.smart_sendto(self.last_pack_to_send)
                cnt = 0
            if not self.pack:
                self.receive_index += self.pack_size + int(length)
            else:
                self.receive_index += self.pack_size + self.max_data_size
            debug(f"{self} now index is {self.receive_index}")

    def send_(self, data: bytes, adding_flag: np.uint32 = Flag.Empty.value, length=0):
        info("Start sending by {}...".format(self))
        flag_val = Flag.Ack.value
        if not self.connected:
            self.make_connection_from_client()
            flag_val |= Flag.Start.value
            info("Connection from client establish")
        self.seq += np.uint(len(data))
        debug(f"{self} update seq: {self.seq}")
        flag_val |= adding_flag
        if adding_flag & Flag.LastPack.value == 0:
            assert self.send_msg(self.make_package(self.seq, self.ack, flag_val, data)) == len(
                data) + self.pack_size
        else:
            assert length
            assert self.send_msg(
                self.make_package(self.seq, self.ack, flag_val, data, length=length)) == len(
                data) + self.pack_size
        return len(data)

    def recv_connection(self):
        while True:
            resp = self.recv_msg(self.pack_size)
            seq, ack, flag, _, data = self.parse_package(resp)
            debug(f"Receive {', '.join(map(str, self.parse_package(resp)))}")
            if seq == 0 and ack == 0 and flag == Flag.Start.value:
                pack = self.make_package(np.uint(1), np.uint(0), Flag.Start.value | Flag.Ack.value,
                                         b"")
                while True:
                    self.smart_sendto(pack)
                    seq, ack, flag, _, data = self.parse_package(
                        self.recv_msg(self.pack_size))
                    if ack == 1 and flag & Flag.Ack.value != 0 and flag & Flag.Start.value != 0:
                        self.connected = True
                        self.seq = self.ack = np.uint(1)
                        self.receive_index -= self.pack_size
                        return
                    if ack != 1:
                        logging.warning(f"{self} get start pack {seq} and {ack}, but wait _ and 1")
                    else:
                        logging.warning(f"{self} flag mismatch on start: {flag}")
            debug(
                f"{self} get start pack {seq}, {ack} and {flag}, but wait 0 and 0 {Flag.Start.value}")

    def get_buffer(self, n: int):
        debug(f"{self} start getting buffer")
        assert n <= 1 << 15
        if len(self.received_buffer) + n + self.pack_size >= self.buffer_size // 2:
            info(f"Buffer resize to {len(self.received_buffer) - self.receive_index}")
            self.received_buffer = self.received_buffer[self.receive_index:]
            self.receive_index = 0
        debug(
            "{} want receive at most {}".format(self, self.buffer_size - len(self.received_buffer)))
        resp = self.recvfrom(self.buffer_size - len(self.received_buffer))
        if len(resp) < 30:
            debug(f"{self} received {resp}")
        else:
            debug(f"{self} received {resp[:30]}...")
        self.received_buffer += resp
        info(f"{self} successful getting of the buffer")

    def recv_msg(self, n: int):
        debug(
            f"{self}: index now is {self.receive_index}."
            f" Length now is {len(self.received_buffer)}.")  # Last message: {self.last_pack}")
        cnt = 0
        while cnt < self.die_cnt:
            if self.receive_index + n <= len(self.received_buffer):
                debug(f"{self} get data from buffer!")
                self.receive_index += n
                debug("Now index of {} is {}".format(self, self.receive_index))
                return self.received_buffer[self.receive_index - n: self.receive_index]
            else:
                try:
                    self.get_buffer(n)
                except TimeoutError:
                    cnt += 1
                    if self.last_pack_to_send is not None:
                        info(f"{self} sent pack again")
                        for _ in range(cnt):
                            self.smart_sendto(self.last_pack_to_send)
                    debug(f"{self} try again to receive")
        return self.make_package(self.seq, self.ack, Flag.End.value, b"")

    def recv_(self, n: int):
        info(f"{self} start receiving")
        while not self.connected:
            self.recv_connection()
            info("Connection from server establish")
        big = False
        if n > self.max_data_size:
            n = len(b"begin")
            big = True
        answer = self.recv_msg(n + self.pack_size)
        seq, ack, flag, length, data = self.parse_package(answer)
        if ack > self.seq:
            logging.warning(f"pack miss {ack} vs {self.seq}")
            return self.recv_(n)
        elif ack < self.seq:
            logging.warning(f"pack duplicated {ack} vs {self.seq}")
            return self.recv_(n)
        else:
            self.ack += np.uint32(len(data))
            debug(f"Update ack: {self.ack}")
            if flag & Flag.PackBegin.value != 0:
                assert big
                info(f"{self} detected big pack")
                raise ValueError
            assert not big
            if flag & Flag.LastPack.value != 0:
                info(f"{self} detected end of big pack. Last pack: {data[:length]}")
                return data[:length]
            return data

    def end_connection(self):
        info(f"{self} finish connection")
        self.seq = np.uint(0)
        self.ack = np.uint(0)
        self.received_buffer = b""
        self.receive_index = 0
        self.connected = False
        self.last_pack_to_send = None
        self.pack = False


class MyTCPProtocol(MyTCPProtocolBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def send(self, data: bytes):
        if len(data) <= self.max_data_size:
            return super().send_(data)
        debug(f"{self} start sending a big package")
        assert super().send_(b"begin", Flag.PackBegin.value) == 5
        assert self.connected
        debug("Message about big pack sent")
        assert super().recv_(2) == b'OK'
        cnt = 0
        info(f"{self} starting big pack sending")
        for i in range(self.max_data_size, len(data), self.max_data_size):
            cnt += self.send_(data[i - self.max_data_size: i])
            self.pack = True  # Now reading only max size packages
            resp = self.recv_(self.max_data_size)
            if resp != data[i - self.max_data_size: i]:
                logging.error(f"\n{resp}" + '\n' + str(data[i - self.max_data_size: i]))
            assert resp == data[i - self.max_data_size: i]
        info("Last pack")
        p = data[cnt:]
        le = len(p)
        p += b'0' * (self.max_data_size - le)
        assert self.send_(p, Flag.LastPack.value, le) == self.max_data_size
        cnt += le
        resp = self.recv_(le)
        assert resp == p[:le]
        info(f"{self} sleep and wait downtime of other({(self.die_cnt + 1) * self.timeout})")
        time.sleep((self.die_cnt + 1) * self.timeout)
        while True:
            try:
                self.recvfrom(le + self.pack_size)
                debug("New mock iteration")
            except TimeoutError:
                break
        info(f"{self} end sending of big pack")
        self.end_connection()
        assert not self.pack
        return cnt

    def recv(self, n: int):
        msg = b''
        try:
            msg = super().recv_(n)
        except ValueError:
            self.send_(b'OK')
            assert self.connected
            info("Starting receiving a big package")
            self.pack = True  # Now reading only a max size packages
            while True:
                cur = super().recv_(self.max_data_size)
                msg += cur
                info(f"Get next pack. Big pack length: {len(msg)}.")
                self.send_(cur)
                if len(cur) != self.max_data_size:
                    info(f"{self} sleep and wait {self.timeout} sec.")
                    time.sleep(self.timeout)
                    info(f"{self} end of receiving big pack")
                    break
        finally:
            assert not self.pack
            return msg
