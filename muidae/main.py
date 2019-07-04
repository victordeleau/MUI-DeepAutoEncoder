from train_autoencoder import train_autoencoder
from tools.parser import parse
from tools.logging import set_logging

if __name__=="__main__":

    args = parse()

    log = set_logging(logging_level=(logging.DEBUG if args.debug else logging.INFO))

    log.info("MUIDAE has started.")

    train_autoencoder(args, display_log=True)