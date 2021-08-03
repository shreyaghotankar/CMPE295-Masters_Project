import React from "react";
import PropTypes from 'prop-types';

import styles from './section.module.scss';

function Section (props) {
    const {
        header,
        side
    } = props;

    return (
        <div>
            {header}
        </div>
    )
}

Section.propTypes = {
    header: PropTypes.string,
    side: PropTypes.string
}

export default Section;