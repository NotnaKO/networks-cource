import logging
import socket
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


class MyTCPProtocol(UDPBasedProtocol):
    # timeout = 5
    timeout = 0.01
    buffer_size = 1 << 13
    pack_size = 24
    die_cnt = 6
    dup_cnt = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq = np.uint(0)
        self.ack = np.uint(0)
        self.received_buffer = b""
        self.receive_index = 0
        self.connected = False
        self.last_pack = None
        self.udp_socket.settimeout(self.timeout)

    @staticmethod
    def make_package(seq: np.uint, ack: np.uint, flag: np.uint32, data: bytes) -> bytes:
        return seq.tobytes() + ack.tobytes() + flag.tobytes() + np.uint32(
            len(data)).tobytes() + data

    @staticmethod
    def parse_package(package: bytes):
        res = np.frombuffer(package, count=2, dtype=np.uint)
        seq = res[0]
        ack = res[1]
        flag, length = np.frombuffer(package, offset=16, count=2, dtype=np.uint32)[0:]
        data = package[MyTCPProtocol.pack_size:]
        debug(f"Get seq={seq}, ack={ack}, flag={flag}, len={length} data={data}")
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
        debug(f"{self} sending {data}")
        self.sendto(data)
        self.last_pack = data
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
                return len(package)
            if ack == self.seq and Flag.Ack.value & flag != 0:
                debug(f"{self} success sent. Ack got.")
                return len(package)
            elif Flag.Ack.value & flag == 0:
                logging.warning(f"Flag mismatched: {flag}")
                raise ValueError
            logging.warning(
                f"{self}: Ack and seq mismatch: {ack} vs {self.seq}. " + "Duplicate"
                if ack < self.seq else "Package reorder")
            cnt += 1
            if cnt == self.dup_cnt:
                info("Sending for duplicating prevent")
                self.smart_sendto(self.last_pack)
                cnt = 0
            self.receive_index += self.pack_size + int(length)

    def send(self, data: bytes):
        return self._send(data)

    def _send(self, data: bytes):
        info("Start sending by {}...".format(self))
        flag_val = Flag.Ack.value
        if not self.connected:
            self.make_connection_from_client()
            flag_val |= Flag.Start.value
            info("Connection from client establish")
        self.seq += np.uint(len(data))
        debug(f"{self} update seq: {self.seq}")
        assert self.send_msg(self.make_package(self.seq, self.ack, flag_val, data)) == len(
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
                    if ack == 1 and flag == Flag.Ack.value | Flag.Start.value:
                        self.connected = True
                        self.seq = self.ack = np.uint(1)
                        self.receive_index -= self.pack_size
                        return
                    debug(f"{self} get start pack {seq} and {ack}, but wait _ and 1")
            debug(f"{self} get start pack {seq} and {ack}, but wait 0 and 0")

    def get_buffer(self, n: int):
        debug(f"{self} start getting buffer")
        assert n <= 1 << 15
        if len(self.received_buffer) + n + self.pack_size >= self.buffer_size:
            info(f"Buffer resize to {self.buffer_size - self.receive_index}")
            self.received_buffer = self.received_buffer[self.receive_index:]
            self.receive_index = 0
        debug(
            "{} want receive at most {}".format(self, self.buffer_size - len(self.received_buffer)))
        resp = self.recvfrom(self.buffer_size - len(self.received_buffer))
        debug(f"{self} received {resp}")
        self.received_buffer += resp
        info(f"{self} successful getting of the buffer")

    def recv_msg(self, n: int):
        debug("{}: index now is {}. Length now is {}. Last message: {}".format(self,
                                                                               self.receive_index,
                                                                               len(self.received_buffer),
                                                                               self.last_pack))
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
                    if self.last_pack is not None:
                        info(f"{self} sent pack again")
                        for _ in range(cnt):
                            self.smart_sendto(self.last_pack)
                    debug(f"{self} try again to receive")
        # self.ending(True)
        # logging.error("End by timeout")
        # exit(0)
        # self.ending(False)
        return self.make_package(self.seq, self.ack, Flag.End.value, b"")

    def _recv(self, n: int):
        info(f"{self} start receiving")
        while not self.connected:
            self.recv_connection()
            info("Connection from server establish")
        answer = self.recv_msg(n + self.pack_size)
        seq, ack, flag, length, data = self.parse_package(answer)
        if ack > self.seq:
            logging.warning(f"pack miss {ack} vs {self.seq}")
            return self.recv(n)
        elif ack < self.seq:
            logging.warning(f"pack duplicated {ack} vs {self.seq}")
            return self.recv(n)
        else:
            self.ack += np.uint32(len(data))
            debug(f"Update ack: {self.ack}")
            return data

    def recv(self, n: int):
        return self._recv(n)
