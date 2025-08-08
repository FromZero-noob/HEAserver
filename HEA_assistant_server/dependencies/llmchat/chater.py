import inspect
import json
import os
import pathlib
import shutil
import tempfile
from typing import Any, Tuple, Union

import pandas as pd

from .log import logger


class PostInitMeta(type):
    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        # 如果有post_init方法，则自动调用
        _post_init = getattr(obj, "_post_init", None)
        if callable(_post_init):
            _post_init()
        return obj


class BaseAssistant(metaclass=PostInitMeta):

    _json_pickle= ["messages", "text_mark"]

    def __repr__(self):
        n_qa  = len([i for i in self.text_mark if i == 4])
        pdr = self.save_msg["project_dir"]
        return f"{self.__class__.__name__}(number_QA={n_qa}, project={pdr}, model={self.model}, root_dir={self.save_msg['root_dir']})"

    def __init__(self,
                 system:str="You are specialist in AI for science.",
                 api_msg:dict=None,
                 offer_engine:str="AZURE",
                 condition:Tuple[str]=(),
                 remove_tmp:bool=False,
                 with_proxy:bool=False,
                 method:str="chat",
                 save_msg:dict=None):
        """
        Parameters
        ----------
        system : str
            System prompt.
        api_msg : dict
            message contrain API key, model name and other information.
        offer_engine : str
            Offer engine name. Options are "AZURE", "OPENAI", "DEEPSEEK" and "GEMINI".
        model : str
            Model name.
        condition : tuple of str
            Condition prompt.
        remove_tmp : bool
            Whether to remove temporary files.
        with_proxy : bool
            Whether to use proxy.
        """

        self.save_msg = save_msg if isinstance(save_msg, dict) else {}
        self.remove_tmp = remove_tmp
        assert method in ["chat", "response"], "Method should be 'chat' or 'response'."
        self.method = method


        if remove_tmp:
            self.remove_old_project_dir()

        if api_msg is None:
            try:
                from llmchat.my_apikey_local import key_dict
            except ImportError:
                from llmchat.read_key import key_dict
        else:
            key_dict = api_msg

        self.api_msg = key_dict

        if offer_engine == "AZURE":
            assert key_dict.get("AZURE_OPENAI_API_KEY",None) is not None, "Please set the AZURE_OPENAI_API_KEY in environment."
            assert key_dict.get("AZURE_API_VERSION",None) is not None, "Please set the AZURE_API_VERSION in environment."
            assert key_dict.get("AZURE_OPENAI_BASE_URL", None) is not None, "Please set the AZURE_OPENAI_BASE_URL in environment."
            assert key_dict.get("AZURE_MODEL_NAME", None) is not None, "Please set the AZURE_MODEL_NAME in environment."

            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                api_key=key_dict.get("AZURE_OPENAI_API_KEY"),
                api_version=key_dict.get("AZURE_API_VERSION"),
                azure_endpoint=key_dict.get("AZURE_OPENAI_BASE_URL")  # Your Azure OpenAI resource's endpoint value.
            )
            self.model = key_dict.get("AZURE_MODEL_NAME")
            if with_proxy:
                self.set_proxy({'HTTP_PROXY': key_dict.get("HTTP_PROXY",None),
                                'HTTPS_PROXY': key_dict.get("HTTPS_PROXY",None)})
            self.dict_format = True
            self.remove_pre_q =False

        elif offer_engine == "OPENAI":
            assert key_dict.get("OPENAI_API_KEY", None) is not None, "Please set the OPENAI_API_KEY in environment."
            assert key_dict.get("OPENAI_MODEL_NAME", None) is not None, "Please set the OPENAI_MODEL in environment."
            from openai import OpenAI
            self.client = OpenAI(
                api_key=key_dict.get("OPENAI_API_KEY"),
            )
            self.model = key_dict.get("OPENAI_MODEL_NAME")
            if with_proxy:
                self.set_proxy({'HTTP_PROXY': key_dict.get("HTTP_PROXY",None),
                                'HTTPS_PROXY': key_dict.get("HTTPS_PROXY",None)})
            self.dict_format = True
            self.remove_pre_q = False

        elif offer_engine == "DEEPSEEK":
            assert key_dict.get("DEEP_SEEK_API_KEY", None) is not None, "Please set the DEEP_SEEK_API_KEY in environment."
            assert key_dict.get("DEEP_SEEK_BASE_URL", None) is not None, "Please set the DEEP_SEEK_BASE_URL in environment."
            from openai import OpenAI
            self.client = OpenAI(
                api_key=key_dict.get("DEEP_SEEK_API_KEY"),
                base_url=key_dict.get("DEEP_SEEK_BASE_URL"),
            )
            self.model = key_dict.get("DEEP_SEEK_MODEL_NAME","deepseek-chat")
            if with_proxy:
                self.set_proxy({'HTTP_PROXY': key_dict.get("HTTP_PROXY",None),
                                'HTTPS_PROXY': key_dict.get("HTTPS_PROXY",None)})
            self.dict_format = True
            self.remove_pre_q = False
        elif offer_engine == "GEMINI":
            assert key_dict.get("GEMINI_API_KEY", None) is not None, "Please set the GEMINI_API_KEY in environment."
            assert key_dict.get("GEMINI_BASE_URL", None) is not None, "Please set the GEMINI_BASE_URL in environment."
            from openai import OpenAI
            self.client = OpenAI(
                api_key=key_dict.get("GEMINI_API_KEY"),
                base_url=key_dict.get("GEMINI_BASE_URL"),
            )
            self.model = key_dict.get("GEMINI_MODEL_NAME","gemini-2.0-flash")
            if with_proxy:
                self.set_proxy({'HTTP_PROXY': key_dict.get("HTTP_PROXY",None),
                                'HTTPS_PROXY': key_dict.get("HTTPS_PROXY",None)})
            self.dict_format = True
            self.remove_pre_q = False

        else:
            raise ValueError("Offer engine not found. Please use 'AZURE' or 'OPENAI','DEEPSEEK'.")

        self.messages = [{"role": "system", "content": system}]
        for i in condition:
            self.messages.append({"role": "user", "content": i})

    def _post_init(self):
        frame = inspect.currentframe()
        a0, a1, a2, input_kv = inspect.getargvalues(frame.f_back)
        # 获取 __init__ 的默认参数字典
        sig = inspect.signature(self.__init__)
        defaults = {
            k: v.default
            for k, v in sig.parameters.items()
            if v.default is not inspect.Parameter.empty and k != "self"
        }
        # 用 input_kv 更新默认参数
        defaults.update(input_kv["kwargs"])

        for k, v in defaults.items():
            setattr(self, k, v)

        self.save_msg = self.save_msg if isinstance(self.save_msg, dict) else {}
        self.process_save_msg()

        if self.remove_tmp:
            self.remove_old_project_dir()

    @classmethod
    def _get_param_names(cls):
        """Get parameter names"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])


    def get_params(self):
        """
        Get parameters for this estimator.


        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            try:
                value = getattr(self, key)
                out[key] = value
            except AttributeError:
                logger.warning(f"The {key} should be defined in the class __init__ method signally, "
                      f"to make it as one attribute.\n"
                      f"such as: self.{key} = {key}")
                pass
        return out

    @property
    def text_mark(self):
        """
        Get the text type mark.
        """
        mark = []

        for i, j in enumerate(self.messages):
            if j.get("role",None) == "system":
                mark.append(0)
            elif j.get("role",None)== "user":
                if "mark" in j and j["mark"] == "long text":
                    mark.append(1)
                else:
                    mark.append(2)
            elif j.get("role",None) == "assistant":
                mark.append(4)
            else:
                mark.append(3)

        return mark


    def process_save_msg(self):

        if "root_dir" not in self.save_msg: # root dir is un-changeable
            self.save_msg["root_dir"]= os.path.join(tempfile.gettempdir(),"llm_pickle_dir")

        if "project_dir" not in self.save_msg:
            self.save_msg["project_dir"]= "file_1"

        if "pdf_name" not in self.save_msg:
            self.save_msg["pdf_name"]= "manuscript_pdf"
        if "sparse_txt_name" not in self.save_msg:
            self.save_msg["sparse_txt_name"]= "manuscript_sparse_txt"
        if "sparse_json_name" not in self.save_msg:
            self.save_msg["sparse_json_name"]= "manuscript_sparse_json"

    def get_temp_save_msg(self, new_save_kwargs=None):
        """
        Get the temp save message.
        """
        if new_save_kwargs is None:
            new_save_kwargs = {}

        if "mn_si_mark" in new_save_kwargs:
            mn_si_mark = new_save_kwargs["mn_si_mark"]
            new_save_kwargs["pdf_name"] = f"{mn_si_mark}_pdf" if "pdf_name" not in new_save_kwargs else new_save_kwargs["pdf_name"]
            new_save_kwargs["sparse_txt_name"] = f"{mn_si_mark}_sparse_txt" if "sparse_txt_name" not in new_save_kwargs else new_save_kwargs["sparse_txt_name"]
            new_save_kwargs["sparse_json_name"] = f"{mn_si_mark}_sparse_json" if "sparse_json_name" not in new_save_kwargs else new_save_kwargs["sparse_json_name"]

        save_msg = self.save_msg.copy()
        save_msg.update(new_save_kwargs)

        return save_msg

    def change_project_dir(self, filename=None, use_last_project_dir=False):
        if use_last_project_dir:
            project_dir = self.save_msg["project_dir"]
        else:
            project_dir = f"{filename}"

        self.save_msg["project_dir"] = project_dir


    def set_proxy(self,proxy:dict=None):
        from openai import DefaultHttpxClient

        default = "None"

        if proxy is None or proxy.get("HTTP_PROXY",None) is None:
            http = os.environ.get('HTTP_PROXY', default).replace("https_proxy=","").replace("https://","").replace("http_proxy=","").replace("http://","")
            https = os.environ.get('HTTPS_PROXY', default).replace("https_proxy=","").replace("https://","").replace("http_proxy=","").replace("http://","")
        else:
            http = proxy.get('HTTP_PROXY', default).replace("https_proxy=","").replace("https://","").replace("http_proxy=","").replace("http://","")
            https = proxy.get('HTTPS_PROXY', default).replace("https_proxy=","").replace("https://","").replace("http_proxy=","").replace("http://","")

        if default not in [http,https]:

            proxy = {
                    "http": f'http://{http}',
                    "https": f'http://{https}',
                }

            http_client=DefaultHttpxClient(proxy=proxy["http"])
            self.client.with_options(http_client=http_client)

            self.client.proxy = {
            "http": os.environ.get('http_proxy', f'https_proxy=http://{http}'),
            "https": os.environ.get('https_proxy', f'https_proxy=http://{https}')
            }

            logger.info(f"Set proxy to {proxy}.")

    def remove_old_project_dir(self):
        try:
            shutil.rmtree(self.save_msg["project_dir"], ignore_errors=True)
        except:
            pass

    def _ask(self,just_last=False, pre_index=None, remove_pre_q=None,
             dict_format=None,**kwargs) -> Union[str, dict]:


        if dict_format is None:
            dict_format = self.dict_format
        if remove_pre_q is None:
            remove_pre_q = self.remove_pre_q

        if just_last:
            msg = [self.messages[-1], ]
        elif pre_index is not None:
            msg = [self.messages[i] for i in pre_index]
        elif remove_pre_q:
            msg = []
            for i, j in enumerate(self.messages):
                if self.text_mark[i] == 4:
                    msg.append(j)
        else:
            msg = self.messages


        if not dict_format: # txt format
            msg = "".join([i["content"] for i in msg])

        if self.client is None:
            answer = self._ask_special(message=msg,text=text,**kwargs)
        else:
            if self.method=="chat":

                response_format = kwargs.pop("response_format", None)
                # response_format={"type": "json_object"}
                chat_completion = self.client.chat.completions.create(model=self.model,
                                                                      messages=msg,
                                                                      response_format=response_format,
                                                                      **kwargs)
                answer = chat_completion.choices[0].message.content

            else:
                text = kwargs.pop("text", None)
                # text={"format": {"type": "json_object"}}
                chat_completion = self.client.responses.create(model=self.model,
                                                                      input=msg,
                                                                      text=text,**kwargs)

                answer = chat_completion.output_text
        return answer

    def read_pdf(self, file_name, sparse_engine="pypdf2", verbose=False, save_to_dir=False,
                 use_last_project_dir=False, tail_ask=True,new_save_kwargs=None,**kwargs):

        file_stem = pathlib.Path(file_name).stem



        if isinstance(sparse_engine, str):
            sparse_engine = sparse_engine.lower()

            if sparse_engine == "pypdf2":
                from llmchat.sparse_pdf import get_pdf_text
                res = get_pdf_text(file_name)

                if verbose:
                    logger.info(f"Read pdf file: {file_name} in fast mode.")

            else:
                from importlib import import_module
                module = import_module("llmchat.sparse_pdf")

                try:
                    read_pdf_txt_other = getattr(module, f"read_pdf_txt_{sparse_engine}")
                except AttributeError:
                    sparse_funcs = [attr for attr in dir(module) if attr.startswith("read_pdf_txt_") and callable(getattr(module, attr))]
                    logger.info(f"Available sparse functions: {sparse_funcs}")
                    raise AttributeError(f"Function read_pdf_txt_{sparse_engine} not found in module {module.__name__}.")

                docx_param_names = inspect.signature(read_pdf_txt_other).parameters
                kw = {k: v for k, v in kwargs.items() if k in docx_param_names}
                res = read_pdf_txt_other(file_name, **kw)

                if verbose:
                    logger.info(f"Read pdf file: {file_name} in {sparse_engine} mode.")

        else:
            docx_param_names = inspect.signature(sparse_engine).parameters
            kw = {k: v for k, v in kwargs.items() if k in docx_param_names}
            res = sparse_engine(file_name, **kw)

            if verbose:
                logger.info(f"Read pdf file: {file_name} by {sparse_engine} function.")

        if isinstance(res, tuple):
            w = str(res[0])
            response = res[1]
        else:
            w = str(res)
            response = {}


        if save_to_dir:

            self.change_project_dir(filename=file_stem, use_last_project_dir=use_last_project_dir)
            temp_save_msg = self.get_temp_save_msg(new_save_kwargs)
            
            txt_dir = os.path.join(temp_save_msg["root_dir"],temp_save_msg["project_dir"], temp_save_msg["sparse_txt_name"])
            json_dir = os.path.join(temp_save_msg["root_dir"],temp_save_msg["project_dir"], temp_save_msg["sparse_json_name"])
            pdf_dir = os.path.join(temp_save_msg["root_dir"],temp_save_msg["project_dir"], temp_save_msg["pdf_name"])
            os.makedirs(txt_dir, exist_ok=True)
            os.makedirs(json_dir, exist_ok=True)

            os.makedirs(pdf_dir, exist_ok=True)

            if pdf_dir != os.path.dirname(file_name):
                try:
                    shutil.copy(file_name, pdf_dir)
                except Exception as e:
                    logger.warning(f"Copy pdf file failed: {e}. Please check the file path and permissions.")

            os.makedirs(txt_dir, exist_ok=True)
            with open(os.path.join(txt_dir, f"{file_stem}.txt"), "w",encoding="utf-8") as f:
                f.write(w)

            os.makedirs(json_dir, exist_ok=True)
            with open(os.path.join(json_dir, f"{file_stem}.json"), "w",encoding="utf-8") as f:
                json.dump(response, f, indent=4, ensure_ascii=False)

            if verbose:

                logger.info(f"Save pdf file to {pdf_dir}, \n    sparse txt file to {txt_dir}, \n    sparse json file to {json_dir}.")

        self.messages.append({"role": "user", "content": w, "mark": "long text"})

        if tail_ask:
            self.messages.append({"role": "user", "content": "Please read the upper words and answer following questions."})

        return w


    def read_pdf_id_token(self, pdf_id_token, verbose=False,save_to_dir=False,use_last_project_dir=False,
                          tail_ask=True,
                          new_save_kwargs=None, **kwargs):

        # Just for debug


        logger.info(f"read_pdf_id_token is Just used for debug, read pdf id token: {pdf_id_token}.")

        from llmchat.sparse_pdf import read_pdf_id_token

        docx_param_names = inspect.signature(read_pdf_id_token).parameters
        kw = {k: v for k, v in kwargs.items() if k in docx_param_names}
        res = read_pdf_id_token(pdf_id_token, **kw)

        if verbose:
            logger.info(f"Read pdf file by id: {pdf_id_token}.")


        if isinstance(res, tuple):
            w = str(res[0])
            response = res[1]
        else:
            w = str(res)
            response = {}

        self.messages.append({"role": "user", "content": w, "mark": "long text"})
        if tail_ask:
            self.messages.append({"role": "user", "content": "Please read the upper words and answer following questions."})

        if save_to_dir:

            self.change_project_dir(filename=pdf_id_token, use_last_project_dir=use_last_project_dir)
            temp_save_msg = self.get_temp_save_msg(new_save_kwargs)

            txt_dir = os.path.join(temp_save_msg["root_dir"],temp_save_msg["project_dir"], temp_save_msg["sparse_txt_name"])
            json_dir = os.path.join(temp_save_msg["root_dir"],temp_save_msg["project_dir"], temp_save_msg["sparse_json_name"])

            os.makedirs(txt_dir, exist_ok=True)
            with open(os.path.join(txt_dir, f"{pdf_id_token}.txt"), "w") as f:
                f.write(w)

            os.makedirs(json_dir, exist_ok=True)
            with open(os.path.join(json_dir, f"{pdf_id_token}.json"), "w") as f:
                json.dump(response, f, indent=4, ensure_ascii=False)
            if verbose:
                logger.info(f"Save sparse txt file to {txt_dir},\n    sparse json file to {json_dir}.")

        return w


    def read_txt(self, file_name: str, verbose=False, save_to_dir=False, use_last_project_dir=False,
                 tail_ask=True,new_save_kwargs=None, **kwargs):


        with open(file_name, "r", encoding="utf-8", errors="ignore") as f:
            w = f.read()
        # 可选：去除不可见或特殊控制字符
        w = ''.join(c for c in w if c.isprintable() or c in '\n\r\t')

        if verbose:
            if len(w) > 30:
                logger.info(f"Read txt file:\n{w[:30]}...")
            else:
                logger.info(f"Read txt file:\n{w}.")

        filename = pathlib.Path(file_name).stem

        self.messages.append({"role": "user", "content": w})

        if tail_ask:
            self.messages.append({"role": "user", "content": "Please read the upper words and answer following questions."})

        if save_to_dir:
            self.change_project_dir(filename=filename, use_last_project_dir=use_last_project_dir)
            temp_save_msg = self.get_temp_save_msg(new_save_kwargs)

            txt_dir = os.path.join(temp_save_msg["root_dir"],temp_save_msg["project_dir"], temp_save_msg["sparse_txt_name"])

            os.makedirs(txt_dir, exist_ok=True)
            with open(os.path.join(txt_dir, f"{filename}.txt"), "w") as f:
                f.write(w)

            if verbose:
                logger.info(f"Save sparse txt file to {txt_dir}.")

        return w

    def read_docx(self, file_name: str, verbose=False, save_to_dir=False, use_last_project_dir=False,
                  tail_ask=True,new_save_kwargs=None, **kwargs):


        from llmchat.sparse_wps import read_doc_docx

        docx_param_names = inspect.signature(read_doc_docx).parameters
        kw = {k: v for k, v in kwargs.items() if k in docx_param_names}
        w = read_doc_docx(file_name, **kw)


        if verbose:
            if len(w) > 30:
                logger.info(f"Read txt file:\n{w[:30]} ...")
            else:
                logger.info(f"Read txt file:\n{w}.")

        filename = pathlib.Path(file_name).stem

        self.messages.append({"role": "user", "content": w})

        if tail_ask:
            self.messages.append({"role": "user", "content": "Please read the upper words and answer following questions."})

        if save_to_dir:

            self.change_project_dir(filename=filename, use_last_project_dir=use_last_project_dir)
            temp_save_msg = self.get_temp_save_msg(new_save_kwargs)

            txt_dir = os.path.join(temp_save_msg["root_dir"],temp_save_msg["project_dir"], temp_save_msg["sparse_txt_name"])
            pdf_dir = os.path.join(temp_save_msg["root_dir"],temp_save_msg["project_dir"], temp_save_msg["pdf_name"])

            os.makedirs(txt_dir, exist_ok=True)
            with open(os.path.join(txt_dir, f"{filename}.txt"), "w") as f:
                f.write(w)

            os.makedirs(pdf_dir, exist_ok=True)
            try:
                os.system(f"libreoffice --headless --convert-to pdf {file_name} --outdir {pdf_dir}")
            except Exception as e:
                logger.error(f"Convert docx to pdf failed: {e}. Please install libreoffice or use other tools to convert docx to pdf.")

            if verbose:
                logger.info(f"Save sparse txt file to {txt_dir}.")


    def read_file(self, file_name: str, **kwargs):
        """
        Read file and return the content.


        Example
        -------
        >>> from llmchat import BaseAssistant
        >>> assistant = BaseAssistant()
        >>> assistant.read_file("test.pdf",sparse_engine="uni-sparser", net_inner=False, save_to_dir=True,
        ...                     new_save_kwargs={"sparse_txt_name":"manuscript_sparse_txt",} )

        Example
        -------
        >>> assistant.read_file("test.pdf", sparse_engine="pypdf2", save_to_dir=True,
        ...                     new_save_kwargs={"sparse_txt_name":"manuscript_sparse_txt_1",} )



        Parameters
        ----------
        file_name : str
            File name.
        verbose : bool
            Whether to print the result.
        save_to_dir : bool
            Whether to save the result to a directory.
        use_last_project_dir : bool
            Whether to use the last project directory.
        tail_ask : bool
            add a tail ask to the result.
        new_save_kwargs : dict
            New save kwargs.
            such as "pdf_name","sparse_txt_name","sparse_json_name"
            eg.
            {"sparse_txt_name":"manuscript_sparse_txt_1",} for sparse txt by pypdf2.
            and {"sparse_txt_name":"manuscript_sparse_txt"} for sparse txt by uni-sparser.
        **kwargs : dict
            Other parameters. such as for read_pdf: "sparse_engine","net_inner"
        """
        if file_name.endswith(".txt"):
            w = self.read_txt(file_name, **kwargs)
        elif file_name.endswith(".pdf"):
            w = self.read_pdf(file_name, **kwargs)
        elif file_name.endswith(".docx") or file_name.endswith(".doc"):
            w = self.read_docx(file_name,  **kwargs)
        elif file_name.endswith(".xlsx") or file_name.endswith(".xls") or file_name.endswith(".csv"):
            w = self.read_excel(file_name,  **kwargs)
        else:
            raise ValueError(f"File type {file_name} not supported.")

        msg = self.messages[-1]

        msg["content"] = msg["content"].replace("'", "`").replace('"', "`").replace(":", "/")

        self.messages[-1] = msg

        return w

    def read_html(self, file_name: str, verbose=False, save_to_dir=False, use_last_project_dir=False,
                  tail_ask=True, new_save_kwargs=None, **kwargs):

        from llmchat.sparse_html import read_html

        docx_param_names = inspect.signature(read_html).parameters
        kw = {k: v for k, v in kwargs.items() if k in docx_param_names}
        w = read_html(file_name, **kw)

        if verbose:
            if len(w) > 30:
                logger.info(f"Read html file:\n{w[:30]} ...")
            else:
                logger.info(f"Read html file:\n{w}.")

        filename = pathlib.Path(file_name).stem

        self.messages.append({"role": "user", "content": w})

        if tail_ask:
            self.messages.append({"role": "user", "content": "Please read the upper words and answer following questions."})

        if save_to_dir:
            self.change_project_dir(filename=filename, use_last_project_dir=use_last_project_dir)
            temp_save_msg = self.get_temp_save_msg(new_save_kwargs)

            txt_dir = os.path.join(temp_save_msg["root_dir"],temp_save_msg["project_dir"], temp_save_msg["sparse_txt_name"])

            os.makedirs(txt_dir, exist_ok=True)
            with open(os.path.join(txt_dir, f"{filename}.txt"), "w") as f:
                f.write(w)

            if verbose:
                logger.info(f"Save sparse txt file to {txt_dir}.")

        return w

    def read_sci_total(self, file_name: str, support_si_dir=None, si_same_dir=False,
                       n_parent=2,remove_dir_name=None, left_dir_name=None,remove_suffix_name=None, left_suffix_name=None,
                       remove_file_mark=None,left_file_mark=None, verbose=False, save_to_dir=False, use_last_project_dir=False,
                       tail_ask=True,new_save_kwargs=None,new_save_kwargs_si=None, **kwargs):

        if remove_file_mark is None:
            remove_file_mark = []

        if left_suffix_name is None:
            left_suffix_name = []

        if remove_suffix_name is None:
            remove_suffix_name = []

        if remove_dir_name is  None:
            remove_dir_name = []

        if left_dir_name is None:
            left_dir_name  = []

        if left_file_mark is None:
            left_file_mark = []

        if new_save_kwargs is None:
            new_save_kwargs = {}

        new_save_kwargs["mn_si_mark"]="manuscript"

        if new_save_kwargs_si is None:
            new_save_kwargs_si = {}

        new_save_kwargs_si["mn_si_mark"]="si"


        if use_last_project_dir is not False:
            logger.warning("For paper + si, in general, suggest use_last_project_dir=False, "
                           "to save all (paper + si) files in the same directory.")

        if si_same_dir:


            support_si_dir = file_name
            for _ in range(n_parent):
                support_si_dir = os.path.dirname(support_si_dir)

            logger.info(f"Use the same directory as {file_name} to read SI files.")

        if support_si_dir is None:

            self.read_file(file_name, verbose=verbose, save_to_dir=save_to_dir,
                              use_last_project_dir=use_last_project_dir, tail_ask=tail_ask,
                              new_save_kwargs=new_save_kwargs, **kwargs)

        else:


            self.read_file(file_name, verbose=verbose, save_to_dir=save_to_dir,
                          use_last_project_dir=False, tail_ask=False,
                          new_save_kwargs=new_save_kwargs, **kwargs)

            # Traverse support_si_dir and its subdirectories to find all file paths containing file_name.stem in their names

            file_stem = pathlib.Path(file_name).stem
            matched_files = []
            for root, dirs, files in os.walk(support_si_dir): # todo
                for fname in files:
                    fname_stem = pathlib.Path(fname).stem
                    if file_stem in str(fname) and fname_stem != file_stem:
                        matched_files.append(os.path.join(root, fname))


            # 1 过滤掉与remove_dir_name相同的文件，防止读取相同的txt文件，如过滤粗解析目录si_sparse_txt，过滤掉原始文件附录文件夹
            index= []
            for indi, i in enumerate(matched_files):
                if any([True if j in remove_dir_name else False for j in pathlib.Path(i).parent.parts]):
                    index.append(indi)
            matched_files = [matched_files[i] for i in range(len(matched_files)) if i not in index]

            if len(left_dir_name) > 0:
                index= []
                for indi, i in enumerate(matched_files):
                    if any([True if j in left_dir_name else False for j in pathlib.Path(i).parent.parts]):
                        index.append(indi)
                matched_files = [matched_files[i] for i in range(len(matched_files)) if i in index]

            # 2 过滤掉与remove_dir_name相同的文件，防止读取相同的txt文件，如过滤粗解析目录si_sparse_txt，过滤掉原始文件附录文件夹
            index= []
            for indi, i in enumerate(matched_files):
                if any([True if i.endswith(j) else False for j in remove_suffix_name]):
                    index.append(indi)
            matched_files = [matched_files[i] for i in range(len(matched_files)) if i not in index]

            if len(left_suffix_name) > 0:
                index= []
                for indi, i in enumerate(matched_files):
                    if any([True if i.endswith(j) else False for j in left_suffix_name]):
                        index.append(indi)
                matched_files = [matched_files[i] for i in range(len(matched_files)) if i in index]

            # 3 过滤掉与remove_file_mark相同的文件，防止读取相同的txt文件，如过滤粗解析目录si_sparse_txt，过滤掉原始文件附录文件夹
            index= []
            for indi, i in enumerate(matched_files):
                if any([True if j in pathlib.Path(i).stem else False  for j in remove_file_mark]):
                    index.append(indi)
            matched_files = [matched_files[i] for i in range(len(matched_files)) if i not in index]

            if len(left_file_mark) > 0:
                index= []
                for indi, i in enumerate(matched_files):
                    if any([True if j in pathlib.Path(i).stem else False  for j in left_file_mark]):
                        index.append(indi)
                matched_files = [matched_files[i] for i in range(len(matched_files)) if i in index]

            # 4 对所有匹配到的文件，若某文件为.txt，则从列表删除同stem的其他格式文件
            prepared = [pathlib.Path(i).stem  for i in matched_files if i.endswith(".txt")]

            index = []
            for indi,i in enumerate(matched_files):
                stem = pathlib.Path(i).stem
                suffix = pathlib.Path(i).suffix
                if stem in prepared and suffix != ".txt":
                    index.append(indi)
            matched_files = [matched_files[i] for i in range(len(matched_files)) if i not in index]


            matched_files = sorted(matched_files)

            logger.info(f"Found files in {support_si_dir} containing '{file_stem}': {matched_files}")

            for i in matched_files:

                if verbose:
                    logger.info(f"Read SI file: {i}.")

                self.read_file(i, verbose=verbose, save_to_dir=save_to_dir,
                              use_last_project_dir=True, tail_ask=tail_ask,
                              new_save_kwargs=new_save_kwargs_si, **kwargs)

    def read_excel(self, file_name: str, verbose=False, save_to_dir=False, use_last_project_dir=False, tail_ask=True,
                   new_save_kwargs=None, **kwargs):


        from llmchat.sparse_wps import read_excel

        docx_param_names = inspect.signature(read_excel).parameters
        kw = {k: v for k, v in kwargs.items() if k in docx_param_names}
        res = read_excel(file_name, **kw)
        w = str(res)

        if verbose:
            if len(w) > 20:
                logger.info(f"Read excel file:\n{w[:20]}...")
            else:
                logger.info(f"Read excel file:\n{w}.")

        filename = pathlib.Path(file_name).stem

        self.messages.append({"role": "user", "content": w})

        if tail_ask:
            self.messages.append({"role": "user", "content": "Please read the upper words and answer following questions."})

        if save_to_dir:
            self.change_project_dir(filename=filename, use_last_project_dir=use_last_project_dir)
            temp_save_msg = self.get_temp_save_msg(new_save_kwargs)

            txt_dir = os.path.join(temp_save_msg["root_dir"],temp_save_msg["project_dir"], temp_save_msg["sparse_txt_name"])
            json_dir = os.path.join(temp_save_msg["root_dir"],temp_save_msg["project_dir"], temp_save_msg["sparse_json_name"])

            os.makedirs(txt_dir, exist_ok=True)


            with open(os.path.join(txt_dir, f"{filename}.txt"), "w") as f:
                f.write(w)

            os.makedirs(json_dir, exist_ok=True)

            with open(os.path.join(json_dir, f"{filename}.json"), "w") as f:
                json.dump(res, f, indent=4, ensure_ascii=False)

            if verbose:
                logger.info(f"Save sparse txt file to {txt_dir}.")

        return w


    def remove_msg(self, index=0):
        if isinstance(index, int):
            indexs = [index]
        else:
            indexs = index
        for index in indexs:
            self.messages.pop(index)
            self.text_mark.pop(index)

    def inquire(self, question, **kwargs):
        return self.Q(question, **kwargs)

    def Q(self, question, just_last=False, verbose=False,
          pre_index=None, remove_pre_q=None,
             dict_format=None,**kwargs):
        """
        Parameters
        ----------
        question : str
            Question to ask.
        just_last : bool
            Whether to use only the last message as input. default is False, use all the messages.
        verbose : bool
            Whether to print the result.
        pre_index : list of int
            Pre-index of the messages to use as input. default is None, use all the messages.
        remove_pre_q : bool
            Whether to remove the previous question. default is False use all the messages.
        dict_format : bool
            Whether to use dictionary format {"role":...} for the input. or just use content text format.
            just used in your own model.
        """

        self.messages.append({"role": "user", "content": str(question)})

        answer = self._ask(just_last=just_last,
                           pre_index=pre_index, remove_pre_q=remove_pre_q,
                           dict_format=dict_format,**kwargs)

        self.messages.append({"role": "assistant", "content": str(answer)})
        if verbose:
            logger.info(f"Result:\n{answer}")


        return answer

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.Q(*args, **kwds)


    def _ask_special(self, message, **kwargs)->str:

        raise NotImplementedError("This method should be implemented in the subclass.")

    def to_json(self, json_name):
        """
        Save the dialogue result to a json file.
        """
        msg = self.get_params()

        os.makedirs(os.path.dirname(json_name), exist_ok=True)

        for i in self._json_pickle:
            if i in self.__dict__:
                msg[i] = self.__dict__[i]
            else:
                msg[i] = getattr(self, i)

        with open(json_name, "w", encoding="utf-8") as f:
            json.dump(msg, f, indent=4, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_name,load_all=False,change_dir_to_file=True,n_parent=2):
        """
        Load the dialogue result from a json file.
        """
        assert os.path.exists(json_name), f"json file {json_name} not found."

        with open(json_name, "r",encoding="utf-8") as f:
            msg = json.load(f)
        res_msg= {}
        add_msg= {}
        init_msg = {}
        init_name = cls._get_param_names()

        for k,v in msg.items():
            if k in cls._json_pickle:
                res_msg[k] = v
            elif k in init_name:
                init_msg[k] = v
            else:
                add_msg[k] = v


        obj = cls(**init_msg)
        for k,v in res_msg.items():
            if k != "text_mark":
                setattr(obj, k, v)
        if load_all:
            for k,v in add_msg.items():
                setattr(obj, k, v)

        if change_dir_to_file:

            pt = pathlib.Path(json_name)
            for _ in range(n_parent):
                pt = pt.parent
            obj.save_msg["project_dir"] = str(pt.name)
            obj.save_msg["root_dir"] = str(pt.parent)
            logger.info(f"Load json file from {json_name}.")
            logger.info(f"Auto change root dir to {str(pt.parent)}, project_dir: {pt.name} , please set change_dir_to_file=False to close auto location.")
        return obj


    def to_loop(self, loop=None, del_old=False):

        project_dir = self.save_msg["project_dir"]
        project_dir = os.path.join(self.save_msg["root_dir"], project_dir)

        project_dir = pathlib.Path(project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)

        if loop is None:

            dirs = [i for i in os.listdir(project_dir) if os.path.isdir(os.path.join(project_dir, i))]

            model_dirs = [i for i in dirs if i.endswith("_model_json")]

            if len(model_dirs) == 0:
                filename = f"0_model_json"
            else:
                model_dirs.sort()
                last_dir = model_dirs[-1]
                last_index = int(last_dir.split("_")[0])
                filename = f"{last_index+1}_model_json"

            if del_old:
                for i in model_dirs:
                    if i != filename:
                        shutil.rmtree(os.path.join(project_dir, i), ignore_errors=True)
                        logger.info(f"Delete old model_json dir: {i}.")
                        dirs.remove(i)

                filename = f"0_model_json"

            json_dir = os.path.join(project_dir, filename)


        else:
            filename = f"{loop}_model_json"
            json_dir = os.path.join(project_dir, filename)

        json_name = os.path.join(json_dir, f"raw.json")

        self.to_json(json_name)

        logger.info(f"Save json model file to {json_name}.")


    def get_res(self, index=-1):
        sel = [n  for n, i in enumerate(self.text_mark) if i ==4 ]
        ind = sel[index]
        return self.messages[ind]["content"]

    def get_res_dict(self):
        sel = [n for n, i in enumerate(self.text_mark) if i ==4 ]
        res = {}
        for n,i in enumerate(sel):
            res[f"res{n+1}"]=self.messages[i]["content"]
        return res

    def to_table(self,*args, **kwargs)-> pd.DataFrame:
        raise NotImplementedError("This method should be implemented in the subclass.")

    def csv_to_loop(self, table:pd.DataFrame=None,mark="_raw",old_mark="",loop=None,**kwargs):

        _ = old_mark

        if table is None:
            df = self.to_table(**kwargs)

        else:
            df = table

        project_dir = self.save_msg["project_dir"]
        project_dir = os.path.join(self.save_msg["root_dir"], project_dir)

        project_dir = pathlib.Path(project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)

        dirs = [i for i in os.listdir(project_dir) if os.path.isdir(os.path.join(project_dir, i))]

        model_dirs = [i for i in dirs if i.endswith("_model_json")]

        if not loop:

            if len(model_dirs) == 0:
                filename = f"0_raw_csv"
            else:
                model_dirs.sort()
                last_dir = model_dirs[-1]
                last_index = int(last_dir.split("_")[0])
                filename = f"{last_index}_raw_csv"
        else:
            filename = f"{loop}_raw_csv"

        csv_dir = os.path.join(project_dir, filename)

        os.makedirs(csv_dir, exist_ok=True)

        csvname = os.path.join(csv_dir, self.save_msg["project_dir"] + mark + ".csv")

        df.to_csv(csvname, index=True, header=True, encoding="utf-8")

        logger.info(f"Save raw csv model file to {csvname}.")

        return csvname


    def change_res(self, res=None, index=-1):

        sel = [n for n, i in enumerate(self.text_mark) if i ==4 ]
        ind = sel[index]

        if isinstance(res, str):
            self.messages[ind]["content"] = res
            self.messages[ind]["role"] = "assistant"
        else:
            self.messages[ind]= res


    def from_loop(self, project_dir=None, loop=None,**kwargs):


        project_dir = pathlib.Path(project_dir)

        dirs = [i for i in os.listdir(project_dir) if os.path.isdir(os.path.join(project_dir, i))]

        model_dirs = [i for i in dirs if i.startswith("_model_json")]

        assert len(model_dirs) > 0, "No ***model_json*** directory found in the project directory."

        if loop is not None:
            filename = f"{loop}_model_json"
        else:
            # load the last model_json_***
            model_dirs.sort()
            last_dir = model_dirs[-1]
            last_index = int(last_dir.split("_")[0])
            filename = f"{last_index}_model_json"

        json_dir = os.path.join(project_dir, filename)

        json_name = os.path.join(json_dir, f"raw.json")

        if not os.path.exists(json_name):
            raise FileNotFoundError(f"json file {json_name} not found.")

        return self.from_json(json_name,**kwargs)







if __name__ == '__main__':
    ba = BaseAssistant(
        offer_engine="AZURE",
        )
    res = ba.Q("openai python api")
    print(res)
