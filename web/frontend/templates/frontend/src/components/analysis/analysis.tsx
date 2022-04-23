import React, {useEffect} from "react";
import {PlatformsFilterT} from "../../types/types";
import axios from "axios";

export function Analysis(props: { filter: PlatformsFilterT[] }) {

  useEffect(() => {
    if (props.filter.filter(f => f.filters.length > 0).length > 0) {
      axios.post('api/analysis', props.filter).then(response => {
        console.log(response.data);
      });
    }
  }, [props.filter]);

  return (
    <>
    </>
  )
}