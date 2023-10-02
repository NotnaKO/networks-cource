import asyncio
import logging
import socket
import time
from enum import Enum
from logging import debug, info

import numpy as np

logging.getLogger().setLevel(level=logging.DEBUG)


class UDPBasedProtocol:
    def __init__(self, *, local_addr, remote_addr):
        self.udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.remote_addr = remote_addr
        self.udp_socket.bind(local_addr)

    def sendto(self, data):
        debug(f"Sending {data}")
        return self.udp_socket.sendto(data, self.remote_addr)

    def recvfrom(self, n):
        debug(f"Try to receiving")
        msg, addr = self.udp_socket.recvfrom(n)

        debug(f"Receiving {msg} from {addr}")
        return msg


class Flag(Enum):
    Empty = np.uint32(0)
    Start = np.uint32(1)
    Ack = np.uint32(2)


class MyTCPProtocol(UDPBasedProtocol):
    timeout = 20
    pack_size = 20

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq = np.uint(0)
        self.ack = np.uint(0)
        self.connected = False

    @staticmethod
    def make_package(seq: np.uint, ack: np.uint, flag: np.uint32, data: bytes) -> bytes:
        return seq.tobytes() + ack.tobytes() + flag.tobytes() + data

    @staticmethod
    def parse_package(package: bytes):
        res = np.frombuffer(package, count=2, dtype=np.uint)
        seq = res[0]
        ack = res[1]
        flag = np.frombuffer(package, offset=16, count=1, dtype=np.uint32)[0]
        data = package[MyTCPProtocol.pack_size:]
        debug(f"Get seq={seq}, ack={ack}, flag={flag}, data={data}")
        return seq, ack, flag, data

    def make_connection_from_client(self):
        while not self.connected:
            self.sendto(self.make_package(np.uint(0), np.uint(0), Flag.Start.value, b""))
            pack = self.recvfrom(self.pack_size)
            seq, ack, flag, data = self.parse_package(pack)
            if seq == 1 and ack == 0 and flag == Flag.Start.value | Flag.Ack.value:
                self.connected = True
                self.sendto(
                    self.make_package(np.uint(1), np.uint(1), Flag.Start.value | Flag.Ack.value,
                                      b""))

                return

    def send(self, data: bytes):
        debug("Start sending...")
        if not self.connected:
            self.make_connection_from_client()
            info("Connection from client establish")
        self.seq += np.uint(len(data))
        debug(f"Update seq: {self.seq}")
        assert self.sendto(
                self.make_package(self.seq, self.ack, Flag.Empty.value, data)) == len(
                data) + self.pack_size
        return len(data)

    def recv_connection(self):
        resp = self.recvfrom(self.pack_size)
        seq, ack, flag, data = self.parse_package(resp)
        debug(f"Receive {', '.join(map(str, self.parse_package(resp)))}")
        if seq == 0 and ack == 0 and flag == Flag.Start.value:
            self.sendto(
                self.make_package(np.uint(1), np.uint(0), Flag.Start.value | Flag.Ack.value, b""))
            seq, ack, flag, data = self.parse_package(self.recvfrom(self.pack_size))
            if seq == ack == 1 and flag == Flag.Start.value | Flag.Ack.value:
                self.connected = True


    def recv(self, n: int):
        debug("Start receiving...")
        if not self.connected:
            self.recv_connection()
            assert self.connected
            info("Connection from server establish")
        answer = self.recvfrom(n + self.pack_size)
        seq, ack, flag, data = self.parse_package(answer)
        if ack != self.seq:
            logging.warning("pack miss")
        assert ack == self.seq
        self.ack += np.uint32(len(data))
        debug(f"Update ack: {self.ack}")
        return data
