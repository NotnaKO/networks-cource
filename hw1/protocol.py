import logging
import socket
from enum import Enum
from logging import debug, info

import numpy as np

logging.getLogger().setLevel(level=logging.WARNING)


class UDPBasedProtocol:
    def __init__(self, *, local_addr, remote_addr):
        self.udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.remote_addr = remote_addr
        self.udp_socket.bind(local_addr)

    def sendto(self, data):
        debug(f"Sending {data}")
        return self.udp_socket.sendto(data, self.remote_addr)

    def recvfrom(self, n):
        debug(f"{self} try to receiving")
        msg, addr = self.udp_socket.recvfrom(n)

        debug(f"{self} receiving {msg} from {addr}")
        return msg


class Flag(Enum):
    Empty = np.uint32(0)
    Start = np.uint32(1)
    Ack = np.uint32(2)
    End = np.uint32(4)


class MyTCPProtocol(UDPBasedProtocol):
    timeout = 0.1
    buffer_size = 1 << 13
    pack_size = 24
    die_cnt = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq = np.uint(0)
        self.ack = np.uint(0)
        self.received_buffer = b""
        self.receive_index = 0
        self.connected = False
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
        debug("Start making connection")
        while not self.connected:
            pack = self.make_package(np.uint(0), np.uint(0), Flag.Start.value, b"")
            self.sendto(pack)
            pack = self.recv_msg(self.pack_size, pack)
            seq, ack, flag, _, data = self.parse_package(pack)
            if seq == 1 and ack == 0 and flag == Flag.Start.value | Flag.Ack.value:
                self.connected = True
                self.sendto(
                    self.make_package(np.uint(1), np.uint(1), Flag.Start.value | Flag.Ack.value,
                                      b""))
                self.seq = self.ack = np.uint(1)
                return

    def send_msg(self, package: bytes):
        ack = self.seq + 1
        while ack != self.seq:
            assert self.sendto(package) == len(package)
            info("Waiting ack by {}".format(self))
            resp = self.recv_msg(self.pack_size, package)
            _, ack, flag, length, _ = self.parse_package(resp)
            self.receive_index -= self.pack_size
            debug("Now index of {} is {}".format(self, self.receive_index))
            if flag == Flag.End.value:
                logging.warning("Ending without ack")
                return len(package)
            if ack == self.seq and Flag.Ack.value & flag != 0:
                break
            elif Flag.Ack.value & flag == 0:
                logging.warning(f"Flag mismatched: {flag}")
                raise ValueError
            info(f"{self} ack got. End of sending message")
            if Flag.End.value & flag != 0:
                self.ending(False)
        return len(package)

    def send(self, data: bytes):
        info("Start sending by {}...".format(self))
        if not self.connected:
            self.make_connection_from_client()
            info("Connection from client establish")
        self.seq += np.uint(len(data))
        debug(f"{self} update seq: {self.seq}")
        assert self.send_msg(self.make_package(self.seq, self.ack, Flag.Ack.value, data)) == len(
            data) + self.pack_size
        return len(data)

    def recv_connection(self):
        resp = self.recv_msg(self.pack_size, None)
        seq, ack, flag, _, data = self.parse_package(resp)
        debug(f"Receive {', '.join(map(str, self.parse_package(resp)))}")
        if seq == 0 and ack == 0 and flag == Flag.Start.value:
            pack = self.make_package(np.uint(1), np.uint(0), Flag.Start.value | Flag.Ack.value, b"")
            self.sendto(pack)
            seq, ack, flag, _, data = self.parse_package(self.recv_msg(self.pack_size, pack))
            if seq == ack == 1 and flag == Flag.Start.value | Flag.Ack.value:
                self.connected = True
            self.seq = self.ack = np.uint(1)

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
        debug(f"{self} successful getting of the buffer")

    def recv_msg(self, n: int, package: bytes | None, ending: bool = True):
        debug("Index now is {}. Length now is {}".format(self.receive_index,
                                                         len(self.received_buffer)))
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
                    if package is not None:
                        self.sendto(package)
                    debug(f"{self} try again to receive")
        # self.ending(True)
        if ending:
            self.ending(False)
        return self.make_package(self.seq, self.ack, Flag.End.value, b"")

    def recv(self, n: int):
        info(f"{self} start receiving")
        if not self.connected:
            self.recv_connection()
            assert self.connected
            info("Connection from server establish")
        answer = self.recv_msg(n + self.pack_size, None)
        seq, ack, flag, _, data = self.parse_package(answer)
        if flag == Flag.End.value:
            assert ack == self.seq
            self.sendto(self.make_package(self.seq, self.ack, Flag.End.value | Flag.Ack.value, b""))
            self.connected = False  # todo: is this right
            self.ending(False)
            return data
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

    def ending(self, wait_ack: bool):
        if self.connected:
            info(f"{self} start ending...")
            if wait_ack:
                self.send_msg(
                    self.make_package(self.seq, self.ack, Flag.End.value | Flag.Ack.value, b""))
            else:
                self.sendto(
                    self.make_package(self.seq, self.ack, Flag.End.value | Flag.Ack.value, b""))
        self.connected = False
        info(f"{self} end work")

    def __del__(self):
        info(f"{self} has been deleted")
        self.ending(True)
