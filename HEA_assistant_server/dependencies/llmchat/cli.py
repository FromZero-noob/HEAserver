import argparse
import os

from llmchat.chater import BaseAssistant
from llmchat.log import logger
from llmchat.set_key import load_env_variable, save_key_in_file


def main():
    parser = argparse.ArgumentParser(description='LLMAssistant.')
    parser.add_argument('action',help='The question to query.',default=None,nargs='*')
    parser.add_argument('-n', action='store_true', help='Start a new session.')
    parser.add_argument('-f',"--file", type=str, help='The input file path. [.pdf, .txt] file is support.',default=None)
    parser.add_argument('-s', "--store_path", type=str, help='file name to store', default="./data/data.json")
    parser.add_argument('-m', "--model", type=str, help='model name', default=None)
    parser.add_argument('-e', "--engine", type=str, help='engine name', default="AZURE")

    args = parser.parse_args()

    if args.n:
        print(">>> Start a new session. <<<")
        assistant = BaseAssistant(offer_engine=args.engine)
    else:
        old = load_env_variable("LLMCHAT_TEMP_PATH")
        if old is not None and os.path.exists(old):
            try:
                print(">>> Continue the session. <<<")
                assistant = BaseAssistant.from_json(old)
            except Exception as e:
                # raise e
                print(">>> Start a new session (unidentifiable checkpoint). <<<")
                assistant = BaseAssistant(offer_engine=args.engine)
        else:
            print(">>> Start a new session (loss old session contact). <<<")
            assistant = BaseAssistant(offer_engine=args.engine)

    if args.file is not None:
        if ".txt" in args.file or ".json" in args.file or ".py" in args.file:
            assistant.read_txt(args.file,verbose=True)
        else:
            assistant.read_pdf(args.file,verbose=True)


    if args.action is None or len(args.action) == 0:
        pass
    else:
        w = assistant.Q(args.action)
        print("Answer:")
        print(w)

    # logger.info(assistant.messages)

    assistant.to_json(args.store_path)

    assistant.remove_old_project_dir()

    abspath = os.path.abspath(args.store_path)

    save_key_in_file("LLMCHAT_TEMP_PATH", str(abspath))

    logger.info(">>> qa '{question}' <<< to add query, or >>> qa '{question}' -n <<<  to start a new session.")

if __name__ == '__main__':
    main()